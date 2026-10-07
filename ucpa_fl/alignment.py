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


def _self_tuned_peer_weights(means, power=2.0):
    """Symmetric self-tuning JSD kernel used by NC-SACPA/SACPA.

    means: [clients, classes, features], assumed normalized per class.
    Returns G [classes, clients, clients] with unit diagonal.
    """
    means=_norm(np.asarray(means,float)); n,C,_=means.shape
    G=np.zeros((C,n,n),float)
    for c in range(C):
        J=np.zeros((n,n),float)
        for i in range(n):
            for k in range(i+1,n):
                J[i,k]=J[k,i]=float(jsd(means[i,c],means[k,c]))
        sig=np.zeros(n,float)
        for i in range(n):
            d=J[i,np.arange(n)!=i]
            sig[i]=float(np.median(d)) if len(d) else 1.0
        np.fill_diagonal(G[c],1.0)
        for i in range(n):
            for k in range(i+1,n):
                scale=np.sqrt(max(sig[i],EPS)*max(sig[k],EPS))
                ratio=J[i,k]/max(scale,EPS)
                val=float(np.exp(-(ratio**power)))
                G[c,i,k]=G[c,k,i]=val
    return G

def sacpa_global_count_prior(means,masks,power=2.0,min_missing_support=2):
    """Phase-VII predecessor: self-tuned peer prior with global support count."""
    means=_norm(np.asarray(means,float)); masks=np.asarray(masks,bool)
    n,C,D=means.shape; G=_self_tuned_peer_weights(means,power); out=np.zeros_like(means); masses=[]
    for c in range(C):
        for i in range(n):
            num=np.zeros(D); den=np.zeros(D)
            for k in range(n):
                if k==i: continue
                w=G[c,i,k]*masks[k,c].astype(float); num+=w*means[k,c]; den+=w
            p=np.divide(num,den,out=np.zeros_like(num),where=den>EPS)
            support=masks[np.arange(n)!=i,c].sum(0)
            block=(~masks[i,c])&(support<int(min_missing_support))
            p[block]=means[i,c,block]
            p[den<=EPS]=means[i,c,den<=EPS]
            if p.sum()<=EPS: p=means[i,c].copy()
            out[i,c]=p/max(p.sum(),EPS); masses.append(float(G[c,i].sum()-1.0))
    return out,{'peer_mass_mean':float(np.mean(masses))}

def nc_sacpa_prior(means,masks,power=2.0,neighborhood_size=3,min_missing_support=2):
    """Neighborhood-Corroborated Scale-Adaptive Peer Alignment prior.

    Missing target coordinates can be imported only when at least
    ``min_missing_support`` of the target's most similar
    ``neighborhood_size`` peers report the coordinate. Imported missing
    coordinates use only that local neighborhood; already-reported target
    coordinates use the full self-tuned peer kernel.
    """
    means=_norm(np.asarray(means,float)); masks=np.asarray(masks,bool)
    n,C,D=means.shape; G=_self_tuned_peer_weights(means,power); out=np.zeros_like(means); masses=[]
    for c in range(C):
        for i in range(n):
            peers=np.array([k for k in np.argsort(-G[c,i]) if k!=i][:min(int(neighborhood_size),max(0,n-1))],int)
            num_all=np.zeros(D); den_all=np.zeros(D)
            for k in range(n):
                if k==i: continue
                w=G[c,i,k]*masks[k,c].astype(float); num_all+=w*means[k,c]; den_all+=w
            p=np.divide(num_all,den_all,out=np.zeros_like(num_all),where=den_all>EPS)
            miss=~masks[i,c]
            allow=np.zeros(D,bool)
            if len(peers): allow=(masks[peers,c].sum(0)>=int(min_missing_support))
            block=miss&(~allow); p[block]=means[i,c,block]
            allowed=miss&allow
            if np.any(allowed):
                num=np.zeros(D); den=np.zeros(D)
                for k in peers:
                    w=G[c,i,k]*masks[k,c].astype(float); num+=w*means[k,c]; den+=w
                loc=np.divide(num,den,out=np.zeros_like(num),where=den>EPS)
                p[allowed]=loc[allowed]
            p[den_all<=EPS]=means[i,c,den_all<=EPS]
            if p.sum()<=EPS: p=means[i,c].copy()
            out[i,c]=p/max(p.sum(),EPS); masses.append(float(G[c,i].sum()-1.0))
    return out,{'peer_mass_mean':float(np.mean(masses))}



def pava_nonincreasing(y):
    """Euclidean projection helper for y_1 >= ... >= y_d using PAVA."""
    y=np.asarray(y,dtype=float)
    blocks=[]
    for i,v in enumerate(y):
        blocks.append([i,i+1,float(v),1.0])
        while len(blocks)>=2 and blocks[-2][2] < blocks[-1][2]-1e-18:
            a,b=blocks[-2],blocks[-1]
            w=a[3]+b[3]
            v=(a[2]*a[3]+b[2]*b[3])/w
            blocks[-2:]=[[a[0],b[1],v,w]]
    out=np.empty_like(y)
    for start,end,val,_ in blocks:
        out[start:end]=val
    return out

def rank_preserving_projection(local, prior):
    """Project a shared prior onto the complete local feature-ranking cone.

    The output is the closest (L2) non-increasing sequence to the prior when
    coordinates are ordered by the local explanation. This preserves the
    client's complete local feature ranking while moving magnitudes as far as
    the ranking constraint allows toward the shared prior.
    """
    e=_norm(np.asarray(local,dtype=float))
    g=_norm(np.asarray(prior,dtype=float))
    if e.ndim!=1 or g.ndim!=1:
        raise ValueError('rank_preserving_projection expects 1D vectors')
    order=np.argsort(-e)
    q=pava_nonincreasing(g[order])
    if len(q)>1:
        eps=max(float(q.max()),1.0)*1e-13
        q=q.copy()
        for j in range(len(q)-2,-1,-1):
            if q[j] <= q[j+1]:
                q[j]=q[j+1]+eps
        q=np.maximum(q,0)
        q=q/(q.sum()+EPS)
    out=np.empty_like(g)
    out[order]=q
    out=np.maximum(out,0)
    total=float(out.sum())
    return out/total if total>0 else np.full_like(out,1.0/len(out))

def rpga_prior(means):
    """RPGA uses the same robust global median artifact prior as xFedAlign."""
    return xfedalign_prior(means)

def make_priors(method,means,var_mean,counts,masks,align_cfg,artifact_cfg,seed=0):
    if method=='local': return means.copy(),{}
    if method=='fedattr_mean': return global_mean_prior(means,counts),{}
    if method=='xfedalign_median': return xfedalign_prior(means),{}
    if method in ('cfba_crc','global_crc'): return xfedalign_prior(means),{'prior':'xfedalign_coordinate_median'}
    if method=='rpga': return rpga_prior(means),{'constraint':'complete_local_ranking'}
    if method=='cluster':
        p,l=cluster_prior(means,align_cfg.cluster_k,seed); return p,{'cluster_labels':l.tolist()}
    if method=='sacpa_global_count':
        return sacpa_global_count_prior(means,masks,power=align_cfg.sacpa_kernel_power,min_missing_support=align_cfg.sacpa_global_min_support)
    if method=='nc_sacpa':
        return nc_sacpa_prior(means,masks,power=align_cfg.ncsacpa_kernel_power,neighborhood_size=align_cfg.ncsacpa_neighborhood_size,min_missing_support=align_cfg.ncsacpa_min_support)
    # attach artifact sparse-gating option dynamically for a compact API
    align_cfg.sparse_intersection_only = artifact_cfg.sparse_intersection_only
    if method=='ucpa': p,e=ucpa_prior(means,var_mean,masks,align_cfg,'full'); return p,{'effective_peers':e.tolist()}
    if method=='ucpa_whole_only': p,e=ucpa_prior(means,var_mean,masks,align_cfg,'whole_only'); return p,{'effective_peers':e.tolist()}
    if method=='ucpa_coord_only': p,e=ucpa_prior(means,var_mean,masks,align_cfg,'coord_only'); return p,{'effective_peers':e.tolist()}
    raise ValueError(method)
