from __future__ import annotations
from dataclasses import dataclass
import numpy as np

EPS=1e-12

@dataclass
class Artifact:
    mean: np.ndarray     # [C,D]
    var_mean: np.ndarray
    counts: np.ndarray
    mask: np.ndarray     # [C,D] coordinates actually transmitted
    bytes_estimate: int


def _quantize(x, bits, lo, hi):
    if bits <= 0 or hi <= lo: return x
    levels=(1<<bits)-1
    q=np.round((np.clip(x,lo,hi)-lo)/(hi-lo)*levels)
    return lo+q/levels*(hi-lo)


def sanitize_artifact(mean, var_mean, counts, cfg, rng: np.random.Generator):
    mean=np.asarray(mean,dtype=np.float64).copy(); var=np.asarray(var_mean,dtype=np.float64).copy()
    C,D=mean.shape; k=min(int(cfg.topk),D)
    out=np.zeros_like(mean); vv=np.zeros_like(var); mask=np.zeros_like(mean,dtype=bool)
    for c in range(C):
        row=np.maximum(mean[c],0)
        norm=np.linalg.norm(row)
        if norm>cfg.clip_radius: row=row*(cfg.clip_radius/(norm+EPS))
        idx=np.argpartition(row,-k)[-k:] if k<D else np.arange(D)
        mask[c,idx]=True
        vals=row[idx]
        hi=max(float(vals.max(initial=0)),EPS)
        vals=_quantize(vals,int(cfg.quant_bits),0.0,hi)
        if cfg.dp_sigma>0:
            vals=vals+rng.normal(0,float(cfg.dp_sigma)*float(cfg.clip_radius)/max(1,k),size=len(vals))
        vals=np.maximum(vals,0)
        out[c,idx]=vals
        vv[c,idx]=np.maximum(var[c,idx],1e-12)
        s=out[c].sum()
        if s>0: out[c]/=s
    # indices + quantized values; variance is stored float32 when enabled.
    index_bytes = 2 if D <= 65535 else 4
    value_bytes = max(1,int(np.ceil(cfg.quant_bits/8)))
    per_coord=index_bytes+value_bytes+(4 if cfg.send_variance else 0)
    bytes_est=int(mask.sum())*per_coord + C*4
    return Artifact(out,vv,counts.copy(),mask,bytes_est)
