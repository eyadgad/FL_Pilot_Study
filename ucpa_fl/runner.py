from __future__ import annotations
from pathlib import Path
import json, traceback
import numpy as np
import torch
from .config import ExperimentConfig, load_config
from .repro import set_seed, resolve_device, environment_snapshot
from .logging_utils import RunLogger
from .datasets import load_data, make_loader
from .federated import train_fedavg, evaluate
from .explain import build_local_explanation
from .artifacts import sanitize_artifact
from .alignment import make_priors, global_mean_prior, xfedalign_prior
from .attacks import apply_artifact_attack
from .metrics import pairwise_edi, reference_edi, summary_jsd, normalize_np, prepare_client_evaluation, evaluate_method_cache
from .privacy import artifact_membership_audit


def _class_hist(ds,n_classes):
    h=np.zeros(n_classes,dtype=int)
    for _,y in make_loader(ds,256,False,0,0):
        for v in torch.as_tensor(y).numpy(): h[int(v)]+=1
    return h.tolist()


def _method_beta(method,cfg):
    if method=='sacpa_global_count': return float(cfg.alignment.sacpa_beta)
    if method=='nc_sacpa': return float(cfg.alignment.ncsacpa_beta)
    return float(cfg.alignment.beta)

def _aligned_summary(method, local_mean, prior, beta):
    if method=='local': return normalize_np(local_mean)
    if method=='fedattr_mean': return normalize_np(prior)
    return normalize_np((1-float(beta))*local_mean+float(beta)*prior)


def _mean_only_bytes(artifact,cfg):
    D=artifact.mean.shape[1]; index_bytes=2 if D<=65535 else 4; value_bytes=max(1,int(np.ceil(cfg.quant_bits/8)))
    return int(artifact.mask.sum())*(index_bytes+value_bytes)+artifact.mean.shape[0]*4


def _evaluate_artifacts(label, artifacts, locals_, eval_caches, oracle_summaries, oracle_counts, task_model, bundle, cfg, seed, device, logger):
    means=np.stack([a.mean for a in artifacts]); vars_=np.stack([a.var_mean for a in artifacts]); counts=np.stack([a.counts for a in artifacts]); masks=np.stack([a.mask for a in artifacts])
    n=len(artifacts); results={}
    for method in cfg.alignment.methods:
        priors,meta=make_priors(method,means,vars_,counts,masks,cfg.alignment,cfg.artifact,seed)
        mbeta=_method_beta(method,cfg)
        summaries=np.stack([_aligned_summary(method,means[i],priors[i],mbeta) for i in range(n)])
        # Method-independent pairwise drift plus method-specific reference EDI.
        p_edi=pairwise_edi(summaries,counts)
        if method=='local': ref=global_mean_prior(means,counts)
        elif method=='fedattr_mean': ref=priors
        elif method=='xfedalign_median': ref=priors
        else: ref=priors
        edi=reference_edi(summaries,ref,counts)
        fidelity=[]
        for i in range(n):
            valid=(counts[i]>0)&(oracle_counts[i]>0)
            v=summary_jsd(summaries[i],oracle_summaries[i],valid)
            if np.isfinite(v): fidelity.append(v)
        per_sample=[]
        for i in range(n):
            pm=evaluate_method_cache(task_model,eval_caches[i],priors[i],method,mbeta,cfg.evaluation,device)
            per_sample.append(pm)
        def avg(key):
            vals=[x[key] for x in per_sample if np.isfinite(x[key])]
            return float(np.mean(vals)) if vals else float('nan')
        if method in ('ucpa','ucpa_whole_only','ucpa_coord_only'):
            comm=float(np.mean([a.bytes_estimate for a in artifacts]))
        elif method=='local': comm=0.0
        else: comm=float(np.mean([_mean_only_bytes(a,cfg.artifact) for a in artifacts]))
        rec={
            'artifact_fidelity_jsd':float(np.mean(fidelity)) if fidelity else float('nan'),
            'sample_fidelity_jsd':avg('sample_fidelity_jsd'),
            'pairwise_edi':p_edi,
            'reference_edi':edi,
            'deletion_auc':avg('deletion_auc'),
            'insertion_auc':avg('insertion_auc'),
            'topk_oracle_overlap':avg('topk_oracle_overlap'),
            'communication_bytes_per_client_artifact':comm,
            'n_eval_explanations':int(sum(x['n_eval_explanations'] for x in per_sample)),
        }
        results[method]=rec
        for k,v in rec.items():
            if isinstance(v,(int,float)) and np.isfinite(v): logger.metric(k,v,stage='explanation',scenario=label,method=method)
        if meta: logger.save_json(f'artifacts/{label}__{method}__meta.json',meta)
    return results


def run_one(cfg: ExperimentConfig, seed: int):
    set_seed(seed,cfg.deterministic); device=resolve_device(cfg.device)
    env=environment_snapshot(); env['resolved_device']=str(device)
    logger=RunLogger(cfg.logging.output_root,cfg.experiment_name,seed,cfg.to_dict(),env)
    logger.event('device_resolved', device=str(device), cuda_available=bool(torch.cuda.is_available()))
    try:
        logger.event('data_loading_started')
        bundle=load_data(cfg,seed)
        logger.save_json('client_partition_summary.json',{
            'client_sizes':bundle.client_sizes,
            'task_class_histograms':[_class_hist(x,bundle.n_classes) for x in bundle.task_clients],
            'surrogate_class_histograms':[_class_hist(x,bundle.n_classes) for x in bundle.surrogate_clients],
            'artifact_class_histograms':[_class_hist(x,bundle.n_classes) for x in bundle.artifact_clients],
            'eval_class_histograms':[_class_hist(x,bundle.n_classes) for x in bundle.eval_clients],
        })
        task_model,task_final=train_fedavg(cfg,bundle,seed,device,logger)
        # Client-held-out task performance captures shifted/non-IID deployment quality.
        client_task=[]
        for cid,ds in enumerate(bundle.eval_clients):
            ev=evaluate(task_model,make_loader(ds,cfg.evaluation.eval_batch_size,False,seed+6000+cid,cfg.num_workers),device)
            client_task.append(ev['accuracy']); logger.metric('client_task_accuracy',ev['accuracy'],stage='final_client',client=cid)
        if client_task:
            logger.metric('client_task_accuracy_mean',float(np.mean(client_task)),stage='final')
            logger.metric('client_task_accuracy_worst',float(np.min(client_task)),stage='final')
        task_final['client_accuracy_mean']=float(np.mean(client_task)) if client_task else float('nan')
        task_final['client_accuracy_worst']=float(np.min(client_task)) if client_task else float('nan')
        # At the final deployed environment, extract explanations from disjoint fit/artifact splits.
        bundle.set_round(cfg.federation.rounds-1)
        locals_=[]; artifacts=[]; rng=np.random.default_rng(seed+70707)
        for cid in range(cfg.federation.n_clients):
            le=build_local_explanation(task_model,bundle.surrogate_clients[cid],bundle.artifact_clients[cid],bundle.input_shape,bundle.n_classes,cfg.surrogate,seed+cid*97,device,cfg.num_workers)
            locals_.append(le); artifacts.append(sanitize_artifact(le.mean,le.var_mean,le.counts,cfg.artifact,rng))
        logger.event('local_artifacts_built',total_bytes=int(sum(a.bytes_estimate for a in artifacts)))
        # Direct privacy audit for UCPA's additional variance channel.
        mia=[]
        for cid in range(cfg.federation.n_clients):
            q=artifact_membership_audit(task_model,locals_[cid],artifacts[cid],bundle.eval_clients[cid],cfg.surrogate.source,bundle.input_shape,bundle.n_classes,device,cfg.surrogate.ig_steps,seed+12000+cid,cfg.num_workers,max_samples=64)
            mia.append(q)
        privacy_summary={}
        for key in ('mean_only_auc','mean_variance_auc'):
            vals=[x[key] for x in mia if np.isfinite(x[key])]
            privacy_summary[key]=float(np.mean(vals)) if vals else float('nan')
            if vals: logger.metric('artifact_mia_'+key,privacy_summary[key],stage='privacy')
        logger.save_json('artifact_membership_audit.json',{'per_client':mia,'summary':privacy_summary})
        if cfg.logging.save_artifacts:
            np.savez_compressed(logger.run_dir/'artifacts'/'clean_artifacts.npz',means=np.stack([a.mean for a in artifacts]),variances=np.stack([a.var_mean for a in artifacts]),counts=np.stack([a.counts for a in artifacts]),masks=np.stack([a.mask for a in artifacts]))

        eval_caches=[];oracle=[];oc=[]
        for cid in range(cfg.federation.n_clients):
            cache=prepare_client_evaluation(task_model,bundle.eval_clients[cid],locals_[cid].surrogate_state,cfg.surrogate.source,bundle.input_shape,bundle.n_classes,device,cfg.surrogate.ig_steps,seed+cid*211,cfg.num_workers,cfg.evaluation.max_eval_samples)
            eval_caches.append(cache);oracle.append(cache['oracle_summary']);oc.append(cache['oracle_counts'])
        oracle=np.stack(oracle);oc=np.stack(oc)
        clean=_evaluate_artifacts('clean',artifacts,locals_,eval_caches,oracle,oc,task_model,bundle,cfg,seed,device,logger)
        attacked=None;bad=[]
        if cfg.attack.enabled:
            attacked_art,bad=apply_artifact_attack(artifacts,cfg.attack,seed+8181)
            attacked=_evaluate_artifacts('attacked',attacked_art,locals_,eval_caches,oracle,oc,task_model,bundle,cfg,seed,device,logger)
            logger.event('artifact_attack_applied',malicious_clients=bad,kind=cfg.attack.kind,strength=cfg.attack.strength)
            # paired deltas within the same trained task model
            for m in clean:
                for metric in ('artifact_fidelity_jsd','pairwise_edi','reference_edi','deletion_auc','insertion_auc','topk_oracle_overlap'):
                    a=attacked[m][metric]; b=clean[m][metric]
                    if np.isfinite(a) and np.isfinite(b): logger.metric(f'delta_{metric}',a-b,stage='attack_delta',method=m)
        final={'seed':seed,'task':task_final,'privacy':privacy_summary,'clean':clean,'attacked':attacked,'malicious_clients':bad}
        logger.save_json('final_results.json',final)
        logger.finalize('completed')
        return logger.run_dir, final
    except Exception as e:
        logger.event('run_exception',error=repr(e),traceback=traceback.format_exc())
        logger.finalize('failed',error=repr(e))
        raise


def run_config(path):
    cfg=load_config(path); out=[]
    for seed in cfg.seeds: out.append(run_one(cfg,int(seed)))
    return out
