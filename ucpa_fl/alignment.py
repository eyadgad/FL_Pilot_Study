from __future__ import annotations
import numpy as np

EPS=1e-12

def _norm(v):
    v=np.maximum(v,0); s=v.sum(-1,keepdims=True); return np.divide(v,s+EPS)

def jsd(p,q):
    p=_norm(np.asarray(p,dtype=float)); q=_norm(np.asarray(q,dtype=float)); m=.5*(p+q)
    def kl(a,b): return np.sum(np.where(a>0,a*np.log((a+EPS)/(b+EPS)),0),axis=-1)
    return .5*kl(p,m)+.5*kl(q,m)

def _weighted_mean(means,weights):
    w=np.asarray(weights,dtype=float); w=w/(w.sum()+EPS)
    return np.tensordot(w,means,axes=(0,0))

def _coord_median(means): return _norm(np.median(means,axis=0))

def _kmeans(x,k,seed=0,steps=30):
    n=len(x); k=min(k,n); rng=np.random.default_rng(seed)
    cent=x[rng.choice(n,k,replace=False)].copy(); lab=np.zeros(n,dtype=int)
    for _ in range(steps):
        d=((x[:,None]-cent[None])**2).sum(-1); nl=d.argmin(1)
        if np.array_equal(nl,lab) and _>0: break
        lab=nl
        for j in range(k):
            if np.any(lab==j): cent[j]=x[lab==j].mean(0)
    return lab

def global_mean_prior(means,counts):
    weights=np.maximum(np.asarray(counts).sum(1),1)
    p=_weighted_mean(means,weights)
    return np.repeat(p[None],len(means),axis=0)

def xfedalign_prior(means):
    p=_coord_median(means)
    return np.repeat(p[None],len(means),axis=0)

def cluster_prior(means,k,seed=0):
    n,C,D=means.shape; x=means.reshape(n,-1); lab=_kmeans(x,k,seed)
    out=np.zeros_like(means)
    for j in np.unique(lab):
        p=_norm(means[lab==j].mean(0)); out[lab==j]=p
    return out,lab

def ucpa_prior(means,var_mean,masks,cfg,mode='full'):
    """Two-scale personalized prior: whole-explanation relevance x coordinate compatibility.

    The sparse mask gates borrowing to coordinates actually reported by both clients.
    Self weight is always one, making the prior well-defined for isolated clients.
    """
    means=np.asarray(means,float); var=np.asarray(var_mean,float); masks=np.asarray(masks,bool)
    n,C,D=means.shape; out=np.zeros_like(means); eff=np.zeros((n,C),float)
    # Per-client variance floor prevents spuriously infinite z-scores from near-zero estimates.
    means=_norm(means)
    positive=var[var>0]
    base=float(np.median(positive)) if positive.size else 1e-12
    # Match the smoke-approved operator exactly: additive variance floor inside sqrt.
    vf=max(EPS,float(cfg.ucpa_variance_floor_fraction)*base)
    for i in range(n):
        for c in range(C):
            num=means[i,c].copy(); den=np.ones(D,float)
            peer_masses=[]
            for k in range(n):
                if k==i: continue
                whole=1.0 if mode=='coord_only' else float(np.exp(-float(jsd(means[i,c],means[k,c]))/max(cfg.ucpa_h,EPS)))
                if mode=='whole_only':
                    coord=np.ones(D,float)
                else:
                    scale=float(cfg.ucpa_z)*np.sqrt(var[i,c]+var[k,c]+vf+EPS)
                    coord=np.exp(-.5*((means[i,c]-means[k,c])/(scale+EPS))**2)
                if getattr(cfg,'sparse_intersection_only',True):
                    coord=coord*(masks[i,c]&masks[k,c])
                w=whole*coord
                num += w*means[k,c]; den += w; peer_masses.append(w.mean())
            out[i,c]=num/(den+EPS); out[i,c]=_norm(out[i,c])
            eff[i,c]=1+sum(peer_masses)
    return out,eff

def make_priors(method,means,var_mean,counts,masks,align_cfg,artifact_cfg,seed=0):
    if method=='local': return means.copy(),{}
    if method=='fedattr_mean': return global_mean_prior(means,counts),{}
    if method=='xfedalign_median': return xfedalign_prior(means),{}
    if method=='cluster':
        p,l=cluster_prior(means,align_cfg.cluster_k,seed); return p,{'cluster_labels':l.tolist()}
    # attach artifact sparse-gating option dynamically for a compact API
    align_cfg.sparse_intersection_only = artifact_cfg.sparse_intersection_only
    if method=='ucpa': p,e=ucpa_prior(means,var_mean,masks,align_cfg,'full'); return p,{'effective_peers':e.tolist()}
    if method=='ucpa_whole_only': p,e=ucpa_prior(means,var_mean,masks,align_cfg,'whole_only'); return p,{'effective_peers':e.tolist()}
    if method=='ucpa_coord_only': p,e=ucpa_prior(means,var_mean,masks,align_cfg,'coord_only'); return p,{'effective_peers':e.tolist()}
    raise ValueError(method)
