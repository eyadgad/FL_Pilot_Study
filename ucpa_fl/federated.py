from __future__ import annotations
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .datasets import DataBundle, make_loader
from .models import build_model


def _weighted_average(states: list[dict], weights: list[float]) -> dict:
    total=float(sum(weights)); w=[x/total for x in weights]
    out={}
    for k in states[0]:
        v0=states[0][k]
        if torch.is_floating_point(v0):
            z=torch.zeros_like(v0)
            for wi,s in zip(w,states): z.add_(s[k], alpha=wi)
            out[k]=z
        else:
            # counters / integer buffers: deterministic copy from largest contributor
            out[k]=states[int(np.argmax(weights))][k].clone()
    return out


def train_local(model: nn.Module, loader, device, epochs: int, lr: float, momentum: float, weight_decay: float):
    model.train(); opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    loss_sum=0.; n=0
    for _ in range(epochs):
        for x,y in loader:
            x=x.to(device); y=torch.as_tensor(y,device=device,dtype=torch.long)
            opt.zero_grad(set_to_none=True); logits=model(x); loss=F.cross_entropy(logits,y); loss.backward(); opt.step()
            loss_sum += float(loss.detach())*len(y); n += len(y)
    return loss_sum/max(1,n)

@torch.no_grad()
def evaluate(model: nn.Module, loader, device):
    model.eval(); correct=0;n=0;loss=0.
    for x,y in loader:
        x=x.to(device); y=torch.as_tensor(y,device=device,dtype=torch.long)
        logits=model(x); loss += float(F.cross_entropy(logits,y,reduction='sum')); correct += int((logits.argmax(1)==y).sum()); n += len(y)
    return {'accuracy': correct/max(1,n), 'loss': loss/max(1,n), 'n': n}


def train_fedavg(cfg, bundle: DataBundle, seed: int, device, logger):
    fed=cfg.federation
    model=build_model(cfg.dataset.name,bundle.input_shape,bundle.n_classes).to(device)
    test_loader=make_loader(bundle.test,cfg.evaluation.eval_batch_size,False,seed+999,cfg.num_workers)
    rng=np.random.default_rng(seed+4242)
    for r in range(fed.rounds):
        bundle.set_round(r)
        m=max(1,int(np.ceil(fed.n_clients*fed.participation_rate)))
        selected=np.sort(rng.choice(fed.n_clients,size=m,replace=False))
        states=[];weights=[];local_losses=[]
        for cid in selected:
            local=deepcopy(model).to(device)
            loader=make_loader(bundle.task_clients[cid],fed.batch_size,True,seed+r*10000+cid,cfg.num_workers)
            ll=train_local(local,loader,device,fed.local_epochs,fed.lr,fed.momentum,fed.weight_decay)
            states.append({k:v.detach().cpu() for k,v in local.state_dict().items()})
            weights.append(len(bundle.task_clients[cid])); local_losses.append(ll)
        model.load_state_dict(_weighted_average(states,weights))
        if cfg.logging.log_every_round:
            ev=evaluate(model,test_loader,device)
            logger.metric('task_accuracy',ev['accuracy'],stage='train',round=r)
            logger.metric('task_loss',ev['loss'],stage='train',round=r)
            logger.metric('local_train_loss_mean',float(np.mean(local_losses)),stage='train',round=r)
            logger.event('round_complete',round=r,selected_clients=[int(x) for x in selected])
    final=evaluate(model,test_loader,device)
    logger.metric('task_accuracy',final['accuracy'],stage='final')
    logger.metric('task_loss',final['loss'],stage='final')
    if cfg.logging.save_checkpoints:
        torch.save(model.state_dict(),logger.run_dir/'checkpoints'/'global_task_model.pt')
    return model, final
