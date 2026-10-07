from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from .alignment import jsd, rank_preserving_projection
from .datasets import make_loader
from .explain import integrated_gradients, normalize_rows, explanation_batch

EPS=1e-12

def normalize_np(x):
    x=np.maximum(np.asarray(x,float),0); return x/(x.sum(-1,keepdims=True)+EPS)

def summary_jsd(a,b,valid=None):
    a=normalize_np(a); b=normalize_np(b)
    vals=[]
    for c in range(a.shape[0]):
        if valid is not None and not valid[c]: continue
        vals.append(float(jsd(a[c],b[c])))
    return float(np.mean(vals)) if vals else float('nan')

def pairwise_edi(summaries,valid_counts):
    n,C,D=summaries.shape; vals=[]
    for c in range(C):
        ids=[i for i in range(n) if valid_counts[i,c]>0]
        for a in range(len(ids)):
            for b in range(a+1,len(ids)):
                vals.append(float(jsd(summaries[ids[a],c],summaries[ids[b],c])))
    return float(np.mean(vals)) if vals else float('nan')

def reference_edi(summaries,reference,valid_counts):
    vals=[]; n,C,_=summaries.shape
    for i in range(n):
        for c in range(C):
            if valid_counts[i,c]>0: vals.append(float(jsd(summaries[i,c],reference[i,c] if reference.ndim==3 else reference[c])))
    return float(np.mean(vals)) if vals else float('nan')

def topk_overlap(a,b,k):
    a=np.asarray(a); b=np.asarray(b); k=min(k,a.size)
    ia=set(np.argpartition(a,-k)[-k:].tolist()); ib=set(np.argpartition(b,-k)[-k:].tolist())
    return len(ia&ib)/max(1,k)


def oracle_client_summary(task_model,dataset,n_classes,input_shape,device,ig_steps=16,max_samples=128,num_workers=0,seed=0):
    loader=make_loader(dataset,64,False,seed,num_workers); buckets=[[] for _ in range(n_classes)]; used=0
    task_model.eval()
    for x,_ in loader:
        x=x.to(device)
        with torch.no_grad(): pred=task_model(x).argmax(1)
        a=integrated_gradients(task_model,x,pred,ig_steps).flatten(1).clamp_min(0)
        a=a/(a.sum(1,keepdim=True)+EPS); a=a.detach().cpu().numpy(); pred=pred.cpu().numpy()
        for j,c in enumerate(pred): buckets[int(c)].append(a[j]); used+=1
        if used>=max_samples: break
    D=int(np.prod(input_shape)); out=np.zeros((n_classes,D)); counts=np.zeros(n_classes,int)
    for c,b in enumerate(buckets):
        if b: out[c]=normalize_np(np.mean(b,0)[None])[0]; counts[c]=len(b)
    return out,counts

@torch.no_grad()
def _prob_target(model,x,target):
    return float(F.softmax(model(x),dim=1)[0,int(target)].detach().cpu())

def perturbation_auc(model,x,target,map_flat,device,steps=20):
    """Input-coordinate deletion/insertion AUC. Lower deletion and higher insertion are better."""
    x=x.to(device); flat=x.flatten().detach(); D=len(flat); order=np.argsort(-np.asarray(map_flat))
    cuts=np.linspace(0,D,steps+1,dtype=int); dele=[]; inse=[]
    for c in cuts:
        idx=torch.as_tensor(order[:c],device=device,dtype=torch.long)
        xd=flat.clone(); xi=torch.zeros_like(flat)
        if c: xd[idx]=0; xi[idx]=flat[idx]
        dele.append(_prob_target(model,xd.view_as(x).unsqueeze(0),target)); inse.append(_prob_target(model,xi.view_as(x).unsqueeze(0),target))
    axis=np.linspace(0,1,steps+1)
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(trapz(dele, axis)), float(trapz(inse, axis))


def evaluate_method_samples(task_model, eval_dataset, surrogate_state, source, prior, method, beta, cfg, input_shape,n_classes,device,seed,num_workers=0,ig_steps=16):
    loader=make_loader(eval_dataset, min(32,cfg.eval_batch_size),False,seed,num_workers)
    jsds=[]; dels=[]; ins=[]; n=0; topovs=[]
    for x,_ in loader:
        if n>=cfg.deletion_insertion_samples_per_client: break
        local,pred=explanation_batch(task_model,surrogate_state,source,x,input_shape,n_classes,device,ig_steps)
        # independent direct-IG oracle for local fidelity
        oracle=integrated_gradients(task_model,x.to(device),pred.to(device),ig_steps).flatten(1).clamp_min(0)
        oracle=oracle/(oracle.sum(1,keepdim=True)+EPS)
        local_np=local.detach().cpu().numpy(); oracle_np=oracle.detach().cpu().numpy(); pred_np=pred.detach().cpu().numpy(); xcpu=x.cpu()
        for j in range(len(x)):
            if n>=cfg.deletion_insertion_samples_per_client: break
            c=int(pred_np[j]); pr=np.asarray(prior[c],float)
            if method=='local': final=local_np[j]
            elif method=='fedattr_mean': final=pr
            elif method=='rpga': final=rank_preserving_projection(local_np[j],pr)
            else: final=normalize_np(((1-beta)*local_np[j]+beta*pr)[None])[0]
            jsds.append(float(jsd(final,oracle_np[j])))
            topovs.append(topk_overlap(final,oracle_np[j],min(cfg.topk_overlap_k,len(final))))
            d,i=perturbation_auc(task_model,xcpu[j],c,final,device,cfg.deletion_steps); dels.append(d); ins.append(i); n+=1
    return {'sample_fidelity_jsd':float(np.mean(jsds)) if jsds else np.nan,
            'deletion_auc':float(np.mean(dels)) if dels else np.nan,
            'insertion_auc':float(np.mean(ins)) if ins else np.nan,
            'topk_oracle_overlap':float(np.mean(topovs)) if topovs else np.nan,
            'n_eval_explanations':n}


def prepare_client_evaluation(task_model, eval_dataset, surrogate_state, source, input_shape, n_classes, device, ig_steps, seed, num_workers=0, max_samples=256):
    """Compute local and direct-IG explanations once; shared by every method."""
    loader=make_loader(eval_dataset,32,False,seed,num_workers); records=[]; buckets=[[] for _ in range(n_classes)]
    for x,_ in loader:
        local,pred=explanation_batch(task_model,surrogate_state,source,x,input_shape,n_classes,device,ig_steps)
        oracle=integrated_gradients(task_model,x.to(device),pred.to(device),ig_steps).flatten(1).clamp_min(0)
        oracle=oracle/(oracle.sum(1,keepdim=True)+EPS)
        local=local.detach().cpu().numpy(); oracle=oracle.detach().cpu().numpy(); pred=pred.detach().cpu().numpy(); x=x.cpu().numpy()
        for j in range(len(pred)):
            c=int(pred[j]); buckets[c].append(oracle[j]); records.append({'x':x[j],'pred':c,'local_map':local[j],'oracle_map':oracle[j]})
            if len(records)>=max_samples: break
        if len(records)>=max_samples: break
    D=int(np.prod(input_shape)); summary=np.zeros((n_classes,D)); counts=np.zeros(n_classes,int)
    for c,b in enumerate(buckets):
        if b: summary[c]=normalize_np(np.mean(b,0)[None])[0]; counts[c]=len(b)
    return {'records':records,'oracle_summary':summary,'oracle_counts':counts}


def evaluate_method_cache(task_model, cache, prior, method, beta, cfg, device):
    jsds=[];dels=[];ins=[];topovs=[]
    records=cache['records'][:int(cfg.deletion_insertion_samples_per_client)]
    for rec in records:
        c=int(rec['pred']);local=np.asarray(rec['local_map']);oracle=np.asarray(rec['oracle_map']);pr=np.asarray(prior[c])
        if method=='local': final=local
        elif method=='fedattr_mean': final=pr
        elif method=='rpga': final=rank_preserving_projection(local,pr)
        else: final=normalize_np(((1-beta)*local+beta*pr)[None])[0]
        jsds.append(float(jsd(final,oracle)));topovs.append(topk_overlap(final,oracle,min(cfg.topk_overlap_k,len(final))))
        x=torch.as_tensor(rec['x'],dtype=torch.float32)
        d,i=perturbation_auc(task_model,x,c,final,device,cfg.deletion_steps);dels.append(d);ins.append(i)
    return {'sample_fidelity_jsd':float(np.mean(jsds)) if jsds else np.nan,'deletion_auc':float(np.mean(dels)) if dels else np.nan,
            'insertion_auc':float(np.mean(ins)) if ins else np.nan,'topk_oracle_overlap':float(np.mean(topovs)) if topovs else np.nan,
            'n_eval_explanations':len(records)}
