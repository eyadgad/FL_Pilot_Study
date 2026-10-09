"""Data-level comparisons, class-summary diagnostic and functional fidelity."""
from __future__ import annotations
import numpy as np
import torch
from study_v7.codec import norm

def jsd(x,y):
    p=norm(np.asarray(x));q=norm(np.asarray(y));m=(p+q)/2
    return float(.5*np.sum(p*np.log((p+1e-12)/(m+1e-12)))+.5*np.sum(q*np.log((q+1e-12)/(m+1e-12))))

def topk(a,b,k):
    aa=np.asarray(a).reshape(-1);bb=np.asarray(b).reshape(-1);k=min(k,len(aa))
    return float(len(set(np.argpartition(aa,-k)[-k:])&set(np.argpartition(bb,-k)[-k:]))/k)

def disagreement(method_maps,classes,nclasses=10):
    """Predicted-class mean-map disagreement; only pair observed classes."""
    summaries=[]
    for p,c in zip(method_maps,classes):
        by={}
        for cl in range(nclasses):
            mask=c==cl
            if mask.any():by[cl]=norm(p[mask].mean(axis=0)[None])[0]
        summaries.append(by)
    vals=[]
    for c in range(nclasses):
        for i in range(len(summaries)):
            for j in range(i+1,len(summaries)):
                if c in summaries[i] and c in summaries[j]:vals.append(jsd(summaries[i][c],summaries[j][c]))
    return float(np.mean(vals)) if vals else float('nan')

def functional_auc(model,x,target,spatial_map,device,steps=20):
    """Delete/insert whole RGB pixel locations together; avoid channel confound.
    Zero baseline and trapezoid of frozen task-model target probability.
    """
    a=x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    if a.ndim!=3:raise ValueError('Need full original C,H,W image')
    C,H,W=a.shape
    ix=np.argsort(-np.asarray(spatial_map).reshape(H*W))
    cutoffs=np.linspace(0,H*W,steps+1,dtype=int)
    dele=[];inse=[]
    for count in cutoffs:
        zeros=np.zeros_like(a);masked=a.copy()
        if count:masked.reshape(C,-1)[:,ix[:count]]=0;zeros.reshape(C,-1)[:,ix[:count]]=a.reshape(C,-1)[:,ix[:count]]
        bat=torch.tensor(np.stack((masked,zeros)),dtype=torch.float32,device=device)
        with torch.no_grad():probs=torch.softmax(model(bat),dim=1)[:,int(target)].cpu().numpy()
        dele.append(float(probs[0]));inse.append(float(probs[1]))
    s=np.linspace(0,1,steps+1)
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(trapz(dele, s)), float(trapz(inse, s))
