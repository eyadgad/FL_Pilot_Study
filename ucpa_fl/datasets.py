from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import math
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset, DataLoader
from torchvision import datasets as tvd, transforms
from torchvision.transforms import functional as TF

from .config import ExperimentConfig


class ClientSubset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int], client_id: int, shift_cfg, round_idx: int = 0, n_clients: int = 8):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.client_id = int(client_id)
        self.shift_cfg = shift_cfg
        self.round_idx = round_idx
        self.n_clients = int(n_clients)

    def set_round(self, r: int):
        self.round_idx = int(r)

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[int(self.indices[i])]
        if not torch.is_tensor(x):
            x = transforms.ToTensor()(x)
        x = self._shift(x, int(y))
        return x, int(y)

    def _shift(self, x: torch.Tensor, y: int) -> torch.Tensor:
        s = self.shift_cfg
        kind = s.kind
        if kind == 'none': return x
        if kind in ('rotation', 'drift_rotation'):
            if kind == 'rotation':
                frac = 0 if self.client_id == 0 else self.client_id / max(1, self.n_clients-1)
                angle = -s.rotation_max_deg + 2*s.rotation_max_deg*frac
            else:
                span = max(1, s.drift_end_round - s.drift_start_round)
                progress = min(1.0, max(0.0, (self.round_idx - s.drift_start_round) / span))
                direction = -1 if self.client_id % 2 == 0 else 1
                angle = direction * s.rotation_max_deg * progress
            return TF.rotate(x, float(angle), interpolation=transforms.InterpolationMode.BILINEAR)
        if kind == 'patch':
            out = x.clone()
            # Client-specific shortcut: only half of labels receive the patch.
            if (y % 2) == (self.client_id % 2):
                h, w = out.shape[-2:]
                ps = min(s.patch_size, h//3, w//3)
                corners = [(1,1), (1,w-ps-1), (h-ps-1,1), (h-ps-1,w-ps-1)]
                r0, c0 = corners[self.client_id % len(corners)]
                out[..., r0:r0+ps, c0:c0+ps] = s.patch_value
            return out
        if kind == 'erasing':
            out = x.clone()
            h,w = out.shape[-2:]
            bh = max(2, int(0.15*h)); bw=max(2,int(0.15*w))
            r0 = (self.client_id * 7) % max(1, h-bh)
            c0 = (self.client_id * 11) % max(1, w-bw)
            out[..., r0:r0+bh, c0:c0+bw] = 0.0
            return out
        if kind == 'color':
            if out_channels(x) != 3: return x
            factor = 0.75 + 0.07*(self.client_id % 8)
            return TF.adjust_contrast(TF.adjust_brightness(x, factor), 2-factor)
        raise ValueError(f'unknown shift kind {kind}')


def out_channels(x): return int(x.shape[-3]) if x.ndim >= 3 else 1


@dataclass
class DataBundle:
    task_clients: list[ClientSubset]
    surrogate_clients: list[ClientSubset]
    artifact_clients: list[ClientSubset]
    eval_clients: list[ClientSubset]
    test: Dataset
    n_classes: int
    input_shape: tuple[int, ...]
    labels: np.ndarray
    client_sizes: list[int]

    def set_round(self, r: int):
        for group in (self.task_clients, self.surrogate_clients, self.artifact_clients, self.eval_clients):
            for ds in group: ds.set_round(r)


def _targets(ds: Dataset) -> np.ndarray:
    if hasattr(ds, 'targets'):
        t = ds.targets
        if torch.is_tensor(t): return t.cpu().numpy().astype(int)
        return np.asarray(t, dtype=int)
    return np.asarray([int(ds[i][1]) for i in range(len(ds))], dtype=int)


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float, seed: int, min_samples: int = 8):
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for attempt in range(100):
        parts = [[] for _ in range(n_clients)]
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
            chunks = np.split(idx, cuts)
            for i, ch in enumerate(chunks): parts[i].extend(ch.tolist())
        if min(map(len, parts)) >= min_samples:
            return [np.asarray(p, dtype=np.int64) for p in parts]
    # fallback: stratified round robin to avoid empty clients
    parts=[[] for _ in range(n_clients)]
    for c in classes:
        idx=np.where(labels==c)[0]; rng.shuffle(idx)
        for j,v in enumerate(idx): parts[j % n_clients].append(int(v))
    return [np.asarray(p,dtype=np.int64) for p in parts]


def iid_partition(n: int, n_clients: int, seed: int):
    rng=np.random.default_rng(seed); idx=np.arange(n); rng.shuffle(idx)
    return [a.astype(np.int64) for a in np.array_split(idx,n_clients)]


def _subsample_size_heterogeneity(parts, sigma: float, seed: int, min_samples: int):
    if sigma <= 0: return parts
    rng=np.random.default_rng(seed+31337)
    raw=rng.lognormal(mean=0.0,sigma=sigma,size=len(parts)); raw/=raw.max()
    out=[]
    for p,f in zip(parts,raw):
        n=max(min_samples, int(len(p)*max(0.15,f)))
        q=p.copy(); rng.shuffle(q); out.append(q[:min(n,len(q))])
    return out


def _split_client_indices(indices: np.ndarray, seed: int):
    rng=np.random.default_rng(seed); idx=indices.copy(); rng.shuffle(idx)
    n=len(idx)
    # Disjoint 65/15/10/10 task/surrogate/artifact/evaluation splits.
    if n < 8:
        # Tiny synthetic sanity fallback; repeated use is recorded by the config's min-sample setting.
        chunks=np.array_split(idx,4); return tuple(np.asarray(c,dtype=np.int64) for c in chunks)
    n_task=max(1,int(0.65*n)); n_surr=max(1,int(0.15*n)); n_art=max(1,int(0.10*n))
    if n_task+n_surr+n_art>=n:
        n_task=max(1,n-3); n_surr=n_art=1
    a=idx[:n_task]; b=idx[n_task:n_task+n_surr]; c=idx[n_task+n_surr:n_task+n_surr+n_art]; d=idx[n_task+n_surr+n_art:]
    return a,b,c,d


def _synthetic(seed: int, n=1600, test_n=400, n_classes=4, shape=(1,16,16)):
    g=torch.Generator().manual_seed(int(seed))
    means=torch.randn(n_classes, int(np.prod(shape)), generator=g)*0.35
    # sparse class-specific regions create explainable structure
    d=int(np.prod(shape))
    for c in range(n_classes): means[c, (c*d//n_classes):((c+1)*d//n_classes)] += 1.5
    y=torch.randint(0,n_classes,(n,),generator=g)
    X=(means[y]+0.8*torch.randn(n,d,generator=g)).reshape(n,*shape)
    yt=torch.randint(0,n_classes,(test_n,),generator=g)
    Xt=(means[yt]+0.8*torch.randn(test_n,d,generator=g)).reshape(test_n,*shape)
    X=(X-X.min())/(X.max()-X.min()+1e-8); Xt=(Xt-Xt.min())/(Xt.max()-Xt.min()+1e-8)
    return TensorDataset(X,y), TensorDataset(Xt,yt), n_classes, shape


def load_data(cfg: ExperimentConfig, seed: int) -> DataBundle:
    name=cfg.dataset.name.lower()
    root=Path(cfg.dataset.root)
    if name=='synthetic':
        train,test,n_classes,shape=_synthetic(seed)
    elif name=='mnist':
        train=tvd.MNIST(root=str(root),train=True,download=cfg.dataset.download,transform=transforms.ToTensor())
        test=tvd.MNIST(root=str(root),train=False,download=cfg.dataset.download,transform=transforms.ToTensor())
        n_classes=10; shape=(1,28,28)
    elif name=='cifar10':
        train=tvd.CIFAR10(root=str(root),train=True,download=cfg.dataset.download,transform=transforms.ToTensor())
        test=tvd.CIFAR10(root=str(root),train=False,download=cfg.dataset.download,transform=transforms.ToTensor())
        n_classes=10; shape=(3,32,32)
    else:
        raise ValueError(f'Unsupported dataset {name}; supported: synthetic,mnist,cifar10')

    if cfg.dataset.train_limit:
        train=Subset(train, list(range(min(len(train),cfg.dataset.train_limit))))
    if cfg.dataset.test_limit:
        test=Subset(test, list(range(min(len(test),cfg.dataset.test_limit))))
    labels=_targets(train)
    fed=cfg.federation
    if fed.partition=='iid': parts=iid_partition(len(train),fed.n_clients,seed)
    elif fed.partition=='dirichlet': parts=dirichlet_partition(labels,fed.n_clients,fed.dirichlet_alpha,seed,fed.min_client_samples)
    else: raise ValueError(f'unknown partition {fed.partition}')
    parts=_subsample_size_heterogeneity(parts,fed.sample_size_sigma,seed,fed.min_client_samples)
    task=[];surr=[];art=[];ev=[]
    for cid,p in enumerate(parts):
        a,b,c,d=_split_client_indices(p,seed+cid*1009)
        task.append(ClientSubset(train,a,cid,cfg.shift,n_clients=fed.n_clients))
        surr.append(ClientSubset(train,b,cid,cfg.shift,n_clients=fed.n_clients))
        art.append(ClientSubset(train,c,cid,cfg.shift,n_clients=fed.n_clients))
        ev.append(ClientSubset(train,d,cid,cfg.shift,n_clients=fed.n_clients))
    return DataBundle(task,surr,art,ev,test,n_classes,shape,labels,[len(p) for p in parts])


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int=0):
    g=torch.Generator().manual_seed(int(seed))
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,generator=g,num_workers=num_workers,drop_last=False)
