"""RWSSA (smoke-frozen rho=0.30) and intentionally simple competing controls.

Important: estimates of direct-IG local-vs-prior errors are sensitive. This code
simulates transmission of dense float32 client gain maps; it is NOT private/DP.
No test/evaluation examples are used for learning gains or selecting lambda.
"""
import numpy as np
from .metrics import normalize_np
from .alignment import xfedalign_prior, jsd
from .risk_control import (crc_upper_empirical_risk,monotone_envelope_nondec,
                           perturbation_auc_many)
from .metrics import topk_overlap
import torch

LAMBDAS=(0.0,0.25,0.50,0.75,1.0)
RHO=0.30

def learn_client_gain(records,prior,n_classes,dim,min_samples=3):
    gain=np.full((n_classes,dim),np.nan,dtype=np.float64)
    counts=np.zeros(n_classes,dtype=np.int32)
    for c in range(n_classes):
        rr=[r for r in records if int(r['pred'])==c]
        counts[c]=len(rr)
        if len(rr)<min_samples: continue
        local=np.stack([normalize_np(np.asarray(r['local_map'])[None])[0] for r in rr])
        oracle=np.stack([normalize_np(np.asarray(r['oracle_map'])[None])[0] for r in rr])
        p=normalize_np(np.asarray(prior[c])[None])[0]
        gain[c]=np.mean(np.abs(local-oracle)-np.abs(p-oracle),axis=0)
    return gain,counts

def global_safety(gains):
    """Smoothed normalized positive gain by class; no class samples -> rho floor.

    Exact smoke rule: 0.25+0.75*(gain-min)/(max-min) on positive gains,
    and zero elsewhere. Finally A = rho+(1-rho)*S.
    """
    gains=np.asarray(gains,float)
    valid=np.isfinite(gains)
    num=np.where(valid,gains,0).sum(axis=0)
    denom=valid.sum(axis=0)
    mean=np.divide(num,denom,out=np.full_like(num,-np.inf),where=denom>0)
    S=np.zeros_like(mean)
    for c in range(mean.shape[0]):
        pos=np.isfinite(mean[c]) & (mean[c]>0)
        if pos.any():
            v=mean[c,pos]; lo=float(v.min()); hi=float(v.max())
            S[c,pos]=0.25+0.75*(v-lo)/(hi-lo+1e-12)
    return S

def weight_map(local,prior,coord_weights,scale):
    l=normalize_np(np.asarray(local)[None])[0]
    p=normalize_np(np.asarray(prior)[None])[0]
    w=np.clip(float(scale)*np.asarray(coord_weights,float),0,1)
    return normalize_np(((1-w)*l+w*p)[None])[0]

def safety_weight(gains,rho=RHO,mode='rwssa'):
    S=global_safety(gains)
    if mode=='rwssa': return rho+(1-rho)*S
    if mode=='binary': return rho+(1-rho)*(S>0)
    if mode=='uniform': return np.ones_like(S)
    if mode=='class_scalar':
        s=np.mean(S,axis=1,keepdims=True)
        return np.broadcast_to(rho+(1-rho)*s,S.shape).copy()
    if mode=='no_safety': return np.ones_like(S)
    raise ValueError(mode)

def select_scale(records, prior, A, *, alpha=0.05, grid=LAMBDAS, topk=32,
                 model=None, device=None, steps=5, risk_mode='jsd_topk'):
    """Selection with a separate calibration set. Implements smoke two-loss
    calibration or four-metric bounded loss. Not a simultaneous high-probability
    guarantee; per-client expected risk correction assumes independent, exchangeable
    calibration records and a fixed safety map learned on other data.
    """
    if not records: return {'lambda':0.0,'upper':[],'n':0,'certified':False}
    L=[]
    for r in records:
        c=int(r['pred'])
        local=normalize_np(np.asarray(r['local_map'])[None])[0]
        oracle=normalize_np(np.asarray(r['oracle_map'])[None])[0]
        p=prior[c]
        maps=np.stack([weight_map(local,p,A[c],lam) for lam in grid])
        ojsd=float(jsd(local,oracle))
        ok=topk_overlap(local,oracle,min(topk,len(local)))
        cols=[np.maximum(0.,np.array([float(jsd(m,oracle)) for m in maps])-ojsd),
              np.maximum(0.,ok-np.array([topk_overlap(m,oracle,min(topk,len(local))) for m in maps]))]
        if risk_mode=='full':
            if model is None: raise ValueError('full risk needs task model')
            d,i=perturbation_auc_many(model,torch.as_tensor(r['x'],dtype=torch.float32),c,maps,device,steps)
            cols += [np.maximum(0,d-d[0]),np.maximum(0,i[0]-i)]
        L.append(np.maximum.reduce(cols))
    L=np.asarray(L,float)
    if not np.isfinite(L).all() or L.min()<-1e-8 or L.max()>1+1e-8:raise ValueError('invalid risk matrix')
    envelope=monotone_envelope_nondec(L)
    upper=crc_upper_empirical_risk(envelope,1.0)
    ok=np.where(upper<=alpha+1e-12)[0]
    idx=int(ok[-1]) if len(ok) else 0
    return {'lambda':float(grid[idx]),'upper':upper.tolist(),'empirical':envelope.mean(0).tolist(),
            'n':int(len(records)),'certified':bool(len(ok)),'risk_mode':risk_mode}

def iflash_style_prior(means,priors,selection_records,temperature=0.1):
    """Faithfulness-weighted client mean PROXY. Not official iFLASH/SHAP.
    Scores local summary JSD against private direct IG on selection records.
    Extra scalar / class transmitted by each client, counted separately.
    """
    n,C,D=means.shape
    scores=np.full((n,C),np.nan,float)
    for i,recs in enumerate(selection_records):
        for c in range(C):
            rr=[r for r in recs if int(r['pred'])==c]
            if len(rr)<3:continue
            scores[i,c]=-np.mean([float(jsd(means[i,c],r['oracle_map'])) for r in rr])
    global_pr=np.zeros((C,D),float)
    for c in range(C):
        valid=np.isfinite(scores[:,c]);
        if valid.any():
            v=scores[valid,c];w=np.exp((v-v.max())/max(temperature,1e-10));w/=w.sum()
            global_pr[c]=np.sum(means[valid,c]*w[:,None],axis=0)
        else: global_pr[c]=priors[0,c]
        global_pr[c]=normalize_np(global_pr[c][None])[0]
    return np.repeat(global_pr[None],n,axis=0), scores

def evaluate_maps(model,records,prior,weights,lam,device,steps,topk):
    """Same held-out records for each method, local-relative four-component loss."""
    from .alignment import jsd
    vals={k:[] for k in ('sample_fidelity_jsd','deletion_auc','insertion_auc','topk_oracle_overlap','excess_fidelity_risk')}
    for r in records:
        c=int(r['pred']); local=normalize_np(np.asarray(r['local_map'])[None])[0]
        oracle=normalize_np(np.asarray(r['oracle_map'])[None])[0]
        m=weight_map(local,prior[c],weights[c],lam)
        j0,j1=float(jsd(local,oracle)),float(jsd(m,oracle))
        k0=topk_overlap(local,oracle,min(topk,len(local)))
        k1=topk_overlap(m,oracle,min(topk,len(local)))
        d,i=perturbation_auc_many(model,torch.as_tensor(r['x'],dtype=torch.float32),c,np.stack([local,m]),device,steps)
        risk=max(0.,float(d[1]-d[0]),float(i[0]-i[1]),float(k0-k1),float(j1-j0))
        for name,v in [('sample_fidelity_jsd',j1),('deletion_auc',d[1]),('insertion_auc',i[1]),('topk_oracle_overlap',k1),('excess_fidelity_risk',risk)]:
            vals[name].append(float(v))
    result={k:float(np.mean(v)) if v else float('nan') for k,v in vals.items()}
    result['excess_fidelity_risk_p90']=float(np.quantile(vals['excess_fidelity_risk'],.9)) if vals['excess_fidelity_risk'] else float('nan')
    result['n_eval_explanations']=len(records)
    return result

def sparse_quantized_client_gains(gains,artifact_masks,bits=8):
    """Exploratory sparse gain channel: reuses already-reported top-k indices.
    Per client/class, clip/quantize signed gains with a transmitted float32 scale.
    Not identical to dense smoke RWSSA and must be validated independently.
    """
    gains=np.asarray(gains,float);masks=np.asarray(artifact_masks,bool)
    if gains.shape!=masks.shape:raise ValueError('mask shape')
    levels=(1<<(bits-1))-1
    out=np.full_like(gains,np.nan)
    N,C,D=gains.shape
    for i in range(N):
        for c in range(C):
            take=masks[i,c] & np.isfinite(gains[i,c])
            if not np.any(take):continue
            vals=gains[i,c,take]
            scale=max(float(np.max(np.abs(vals))),1e-12)
            out[i,c,take]=np.rint(np.clip(vals/scale,-1,1)*levels)/levels*scale
    bytes_per_client=int(masks[0].sum())*(bits//8)+C*4
    return out, bytes_per_client
