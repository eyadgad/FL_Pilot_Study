from __future__ import annotations
import numpy as np
import torch
from .alignment import jsd
from .datasets import make_loader
from .explain import explanation_batch

EPS=1e-12

def auc_rank(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    if len(pos)==0 or len(neg)==0:return float('nan')
    # exact pairwise AUC; small privacy audit sample sizes keep this cheap.
    wins=0.; total=0
    for i in pos:
        for j in neg:
            wins += 1.0 if s[i]>s[j] else (0.5 if s[i]==s[j] else 0.0); total+=1
    return wins/total

def artifact_membership_audit(task_model, local_explanation, transmitted_artifact, eval_dataset, source, input_shape,n_classes,device,ig_steps,seed,num_workers=0,max_samples=64):
    member=local_explanation.sample_maps[:max_samples]
    loader=make_loader(eval_dataset,32,False,seed,num_workers); non=[]
    for x,_ in loader:
        maps,pred=explanation_batch(task_model,local_explanation.surrogate_state,source,x,input_shape,n_classes,device,ig_steps)
        maps=maps.detach().cpu().numpy(); pred=pred.detach().cpu().numpy()
        for j in range(len(x)):
            non.append({'local_map':maps[j],'pred':int(pred[j])})
            if len(non)>=max_samples: break
        if len(non)>=max_samples: break
    y=[]; sm=[]; sv=[]
    posvar=transmitted_artifact.var_mean[transmitted_artifact.var_mean>0]
    vf=max(float(np.median(posvar))*0.05 if posvar.size else 1e-8,1e-10)
    for label,items in [(1,member),(0,non)]:
        for rec in items:
            c=int(rec['pred']); a=np.asarray(rec['local_map'],float); mu=transmitted_artifact.mean[c]; mask=transmitted_artifact.mask[c]
            if not np.any(mask): continue
            y.append(label); sm.append(-float(jsd(a,mu)))
            v=np.maximum(transmitted_artifact.var_mean[c],vf)
            sv.append(-float(np.mean(((a[mask]-mu[mask])**2)/(v[mask]+EPS))))
    return {'mean_only_auc':auc_rank(y,sm),'mean_variance_auc':auc_rank(y,sv),'n_member':int(sum(np.asarray(y)==1)),'n_nonmember':int(sum(np.asarray(y)==0))}
