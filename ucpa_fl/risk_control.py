from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .alignment import jsd
from .metrics import normalize_np, topk_overlap

EPS = 1e-12


def monotone_envelope_nondec(losses: np.ndarray) -> np.ndarray:
    """Cumulative-max envelope along the last axis (increasing alignment)."""
    x = np.asarray(losses, dtype=float)
    return np.maximum.accumulate(x, axis=-1)


def crc_upper_empirical_risk(losses: np.ndarray, bound: float = 1.0) -> np.ndarray:
    """Finite-sample CRC correction for bounded monotone losses.

    For losses in [0, bound], returns n/(n+1) * empirical risk + bound/(n+1).
    The alignment grid is assumed ordered from weakest to strongest, with a
    nondecreasing loss after monotonization. This is the standard CRC formula
    applied to lambda = 1 - beta.
    """
    x = np.asarray(losses, dtype=float)
    if x.ndim != 2:
        raise ValueError('losses must be [n_calibration, n_grid]')
    n = x.shape[0]
    if n < 1:
        raise ValueError('at least one calibration example is required')
    return (n / (n + 1.0)) * x.mean(axis=0) + float(bound) / (n + 1.0)


def choose_largest_certified_beta(raw_losses: np.ndarray, beta_grid, alpha: float, bound: float = 1.0):
    """Return largest beta whose monotonized CRC risk bound is <= alpha."""
    beta = np.asarray(beta_grid, dtype=float)
    if np.any(np.diff(beta) < 0):
        raise ValueError('beta_grid must be sorted ascending')
    raw = np.asarray(raw_losses, dtype=float)
    if raw.shape[1] != len(beta):
        raise ValueError('loss matrix/grid mismatch')
    env = monotone_envelope_nondec(raw)
    upper = crc_upper_empirical_risk(env, bound=bound)
    ok = np.where(upper <= float(alpha) + 1e-15)[0]
    idx = int(ok[-1]) if len(ok) else 0
    # beta=0 is a local/no-alignment fallback; it should have exactly zero
    # excess loss by construction, but CRC finite-sample correction can exceed
    # alpha for tiny n. We still return beta=0 as the safe reference action and
    # mark certified=False when even the CRC bound cannot certify it.
    return {
        'beta': float(beta[idx]) if len(ok) else float(beta[0]),
        'index': idx if len(ok) else 0,
        'certified': bool(len(ok)),
        'upper_risk': upper.tolist(),
        'empirical_risk': env.mean(axis=0).tolist(),
        'n_calibration': int(raw.shape[0]),
    }


def aligned_map(local_map, prior_map, beta: float):
    return normalize_np(((1.0 - float(beta)) * np.asarray(local_map) + float(beta) * np.asarray(prior_map))[None])[0]


def _target_probs_many(model, batch: torch.Tensor, target: int, device):
    with torch.no_grad():
        logits = model(batch.to(device))
        return F.softmax(logits, dim=1)[:, int(target)].detach().cpu().numpy()


def perturbation_auc_many(model, x: torch.Tensor, target: int, maps: np.ndarray, device, steps: int = 8):
    """Vectorized deletion/insertion AUC for several attribution maps."""
    maps = np.asarray(maps, dtype=float)
    if maps.ndim == 1:
        maps = maps[None]
    flat = x.detach().flatten().cpu()
    D = flat.numel()
    cuts = np.linspace(0, D, int(steps) + 1, dtype=int)
    variants = []
    meta = []
    for mi, m in enumerate(maps):
        order = np.argsort(-m)
        for ci, c in enumerate(cuts):
            idx = torch.as_tensor(order[:c], dtype=torch.long)
            xd = flat.clone()
            xi = torch.zeros_like(flat)
            if c:
                xd[idx] = 0
                xi[idx] = flat[idx]
            variants.extend([xd, xi])
            meta.extend([(mi, ci, 0), (mi, ci, 1)])
    batch = torch.stack(variants).reshape(len(variants), *x.shape)
    probs = _target_probs_many(model, batch, target, device)
    dele = np.zeros((len(maps), len(cuts)), float)
    inse = np.zeros_like(dele)
    for p, (mi, ci, typ) in zip(probs, meta):
        (dele if typ == 0 else inse)[mi, ci] = float(p)
    axis = np.linspace(0, 1, len(cuts))
    # np.trapezoid is the NumPy 2.x spelling.
    return np.trapezoid(dele, x=axis, axis=1), np.trapezoid(inse, x=axis, axis=1)


def calibration_loss_matrix(model, records, prior, beta_grid, device, steps: int, topk: int):
    """Bounded excess-fidelity loss against direct task-model IG.

    L = max(0,
            deletion(beta)-deletion(local),
            insertion(local)-insertion(beta),
            oracle_topk(local)-oracle_topk(beta)).
    Each component lies in [0,1], hence L in [0,1].
    """
    beta = np.asarray(beta_grid, dtype=float)
    rows = []
    diagnostics = []
    for rec in records:
        c = int(rec['pred'])
        local = normalize_np(np.asarray(rec['local_map'])[None])[0]
        oracle = normalize_np(np.asarray(rec['oracle_map'])[None])[0]
        pr = normalize_np(np.asarray(prior[c])[None])[0]
        maps = np.stack([aligned_map(local, pr, b) for b in beta])
        x = torch.as_tensor(rec['x'], dtype=torch.float32)
        dels, ins = perturbation_auc_many(model, x, c, maps, device, steps)
        kvals = np.array([topk_overlap(m, oracle, min(int(topk), len(m))) for m in maps], float)
        jvals = np.array([float(jsd(m, oracle)) for m in maps], float)
        d0, i0, k0, j0 = float(dels[0]), float(ins[0]), float(kvals[0]), float(jvals[0])
        harm = np.maximum.reduce([
            np.zeros(len(beta), float),
            np.clip(dels - d0, 0, 1),
            np.clip(i0 - ins, 0, 1),
            np.clip(k0 - kvals, 0, 1),
            np.clip(jvals - j0, 0, 1),
        ])
        rows.append(harm)
        diagnostics.append({'deletion': dels.tolist(), 'insertion': ins.tolist(), 'topk': kvals.tolist(), 'jsd': jvals.tolist()})
    return np.asarray(rows, float), diagnostics


def evaluate_beta_on_records(model, records, prior, beta: float, device, steps: int, topk: int):
    jsds=[]; dels=[]; ins=[]; tops=[]; harms=[]
    for rec in records:
        c=int(rec['pred'])
        local=normalize_np(np.asarray(rec['local_map'])[None])[0]
        oracle=normalize_np(np.asarray(rec['oracle_map'])[None])[0]
        pr=normalize_np(np.asarray(prior[c])[None])[0]
        final=aligned_map(local,pr,beta)
        j0=float(jsd(local,oracle)); j1=float(jsd(final,oracle)); jsds.append(j1)
        k0=float(topk_overlap(local,oracle,min(int(topk),len(local))))
        k1=float(topk_overlap(final,oracle,min(int(topk),len(final))))
        tops.append(k1)
        x=torch.as_tensor(rec['x'],dtype=torch.float32)
        d,i=perturbation_auc_many(model,x,c,np.stack([local,final]),device,steps)
        d0,d1=float(d[0]),float(d[1]); i0,i1=float(i[0]),float(i[1])
        dels.append(d1); ins.append(i1)
        harms.append(max(0.0,d1-d0,i0-i1,k0-k1,j1-j0))
    return {
        'sample_fidelity_jsd':float(np.mean(jsds)) if jsds else np.nan,
        'deletion_auc':float(np.mean(dels)) if dels else np.nan,
        'insertion_auc':float(np.mean(ins)) if ins else np.nan,
        'topk_oracle_overlap':float(np.mean(tops)) if tops else np.nan,
        'excess_fidelity_risk':float(np.mean(harms)) if harms else np.nan,
        'excess_fidelity_risk_p90':float(np.quantile(harms,0.9)) if harms else np.nan,
        'n':len(records),
    }

def optimize_betas_under_caps(means, counts, prior, beta_grid, caps, pairwise_edi_fn, max_passes: int = 8):
    """Coordinate-descent drift minimization subject to per-client certified caps.

    Uses only transmitted class-summary artifacts. Any chosen beta_i <= cap_i, so
    monotone conformal risk control for client i remains valid.
    """
    means=np.asarray(means,float); counts=np.asarray(counts)
    grid=np.asarray(beta_grid,float); caps=np.asarray(caps,float)
    n=len(caps)
    allowed=[grid[grid <= caps[i] + 1e-12] for i in range(n)]
    def score(betas):
        summaries=np.stack([normalize_np(((1-betas[i])*means[i] + betas[i]*prior[i])) for i in range(n)])
        return float(pairwise_edi_fn(summaries,counts))
    starts=[np.zeros(n,float), np.full(n,float(np.min(caps))), caps.copy()]
    best_b=None;best_s=float('inf')
    for start in starts:
        b=np.array([max([x for x in allowed[i] if x <= start[i]+1e-12], default=0.0) for i in range(n)],float)
        for _ in range(max_passes):
            changed=False
            for i in range(n):
                cur=b[i]; local_best=score(b); local_val=cur
                for v in allowed[i]:
                    b[i]=float(v); s=score(b)
                    if s < local_best - 1e-12 or (abs(s-local_best)<=1e-12 and v < local_val):
                        local_best=s;local_val=float(v)
                b[i]=local_val
                changed |= abs(local_val-cur)>1e-12
            if not changed: break
        s=score(b)
        if s<best_s: best_s=s;best_b=b.copy()
    return best_b.tolist(),best_s
