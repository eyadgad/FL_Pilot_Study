from __future__ import annotations
import numpy as np
from .artifacts import Artifact

def apply_artifact_attack(artifacts: list[Artifact], cfg, seed: int):
    if not cfg.enabled or cfg.fraction<=0: return artifacts,[]
    rng=np.random.default_rng(seed); n=len(artifacts); m=max(1,int(round(n*cfg.fraction)))
    bad=sorted(rng.choice(n,m,replace=False).tolist()); out=[]
    for i,a in enumerate(artifacts):
        if i not in bad: out.append(a); continue
        mean=a.mean.copy(); mask=a.mask.copy(); var=a.var_mean.copy(); C,D=mean.shape
        if cfg.kind=='artifact_shift':
            mean=np.roll(mean,max(1,int(cfg.strength*D)),axis=1)
            mask=np.roll(mask,max(1,int(cfg.strength*D)),axis=1); var=np.roll(var,max(1,int(cfg.strength*D)),axis=1)
        elif cfg.kind=='random_support':
            for c in range(C):
                idx=rng.choice(D,max(1,int(mask[c].sum())),replace=False); mean[c]=0; mask[c]=False
                vals=rng.random(len(idx)); vals/=vals.sum(); mean[c,idx]=vals; mask[c,idx]=True; var[c,idx]=1e-6
        else: raise ValueError(cfg.kind)
        out.append(Artifact(mean,var,a.counts.copy(),mask,a.bytes_estimate))
    return out,bad
