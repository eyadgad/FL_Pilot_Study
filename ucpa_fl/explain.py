from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from .datasets import make_loader
from .surrogate import fit_surrogate

EPS=1e-12


def normalize_rows(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    x = torch.clamp(x, min=0)
    s = x.flatten(1).sum(1).view(-1, *([1]*(x.ndim-1)))
    return x / (s + eps)


def integrated_gradients(model, x: torch.Tensor, targets: torch.Tensor, steps: int = 16, baseline=None):
    """Batched IG with a zero baseline. Returns absolute pixel/channel attributions."""
    model.eval()
    if baseline is None: baseline = torch.zeros_like(x)
    total_grad = torch.zeros_like(x)
    # midpoint rule is cheaper and avoids endpoint instability.
    alphas = (torch.arange(steps, device=x.device, dtype=x.dtype) + 0.5) / steps
    for a in alphas:
        z = (baseline + a * (x-baseline)).detach().requires_grad_(True)
        logits = model(z)
        score = logits.gather(1, targets[:,None]).sum()
        grad, = torch.autograd.grad(score, z, retain_graph=False, create_graph=False)
        total_grad += grad.detach()
    return ((x-baseline) * total_grad / steps).abs()


@dataclass
class LocalExplanation:
    mean: np.ndarray       # [C,D]
    var_mean: np.ndarray   # [C,D] estimated variance of sample mean
    counts: np.ndarray     # [C]
    sample_maps: list      # list of dicts for evaluation
    surrogate_state: dict | None = None


def _flatten_norm(attr: torch.Tensor) -> torch.Tensor:
    a = attr.flatten(1).clamp_min(0)
    return a / (a.sum(1, keepdim=True) + EPS)


def build_local_explanation(task_model, fit_dataset, artifact_dataset, input_shape, n_classes: int, surrogate_cfg, seed: int, device, num_workers: int = 0):
    source = surrogate_cfg.source
    surrogate = None
    if source == 'linear_surrogate':
        surrogate = fit_surrogate(task_model, fit_dataset, input_shape, n_classes, surrogate_cfg, seed, device, num_workers)
    elif source != 'task_ig':
        raise ValueError(f'unknown explanation source {source}')

    loader = make_loader(artifact_dataset, 64, False, seed+17, num_workers)
    buckets=[[] for _ in range(n_classes)]
    samples=[]
    max_per = int(surrogate_cfg.artifact_samples_per_class)
    task_model.eval()
    for x,y in loader:
        x=x.to(device)
        with torch.no_grad(): pred=task_model(x).argmax(1)
        if source=='linear_surrogate':
            attr=surrogate.attribution(x,pred)
        else:
            attr=integrated_gradients(task_model,x,pred,int(surrogate_cfg.ig_steps))
        af=_flatten_norm(attr).detach().cpu()
        xp=x.detach().cpu(); yp=torch.as_tensor(y).cpu(); pp=pred.detach().cpu()
        for j in range(len(x)):
            c=int(pp[j])
            if len(buckets[c]) < max_per:
                buckets[c].append(af[j].numpy())
            # Keep a bounded evaluation sample cache; caller can sub-sample further.
            if len(samples) < n_classes * int(surrogate_cfg.eval_samples_per_class):
                samples.append({'x':xp[j].numpy(), 'y':int(yp[j]), 'pred':c, 'local_map':af[j].numpy()})
        if all(len(b)>=max_per for b in buckets): break

    D=int(np.prod(input_shape))
    mean=np.zeros((n_classes,D),dtype=np.float64)
    var=np.zeros_like(mean)
    counts=np.zeros(n_classes,dtype=np.int64)
    for c,b in enumerate(buckets):
        if not b: continue
        a=np.stack(b).astype(np.float64); counts[c]=len(a); mean[c]=a.mean(0)
        if len(a)>1: var[c]=a.var(0,ddof=1)/len(a)
        else: var[c]=np.maximum(mean[c]*(1-mean[c]),1e-6)
    state={k:v.detach().cpu() for k,v in surrogate.state_dict().items()} if surrogate is not None else None
    return LocalExplanation(mean,var,counts,samples,state)


def direct_ig_maps(task_model, xs: torch.Tensor, device, steps: int=16):
    xs=xs.to(device); task_model.eval()
    with torch.no_grad(): pred=task_model(xs).argmax(1)
    a=integrated_gradients(task_model,xs,pred,steps)
    return _flatten_norm(a).detach().cpu().numpy(), pred.detach().cpu().numpy()


def load_surrogate_from_state(input_shape, n_classes, state, device):
    from .surrogate import SparseLinearSurrogate
    s=SparseLinearSurrogate(input_shape,n_classes).to(device)
    s.load_state_dict(state); s.eval(); return s


def explanation_batch(task_model, surrogate_state, source: str, x: torch.Tensor, input_shape, n_classes: int, device, ig_steps: int):
    x=x.to(device); task_model.eval()
    with torch.no_grad(): pred=task_model(x).argmax(1)
    if source=='linear_surrogate':
        s=load_surrogate_from_state(input_shape,n_classes,surrogate_state,device)
        attr=s.attribution(x,pred)
    elif source=='task_ig':
        attr=integrated_gradients(task_model,x,pred,ig_steps)
    else: raise ValueError(source)
    return _flatten_norm(attr), pred
