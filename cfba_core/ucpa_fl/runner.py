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
from .alignment import make_priors, global_mean_prior, xfedalign_prior, rank_preserving_projection
from .attacks import apply_artifact_attack
from .metrics import pairwise_edi, reference_edi, summary_jsd, normalize_np, prepare_client_evaluation, evaluate_method_cache
from .risk_control import calibration_loss_matrix, choose_largest_certified_beta, evaluate_beta_on_records
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
    if method=='rpga':
        return np.stack([rank_preserving_projection(local_mean[c],prior[c]) for c in range(local_mean.shape[0])])
    return normalize_np((1-float(beta))*local_mean+float(beta)*prior)


def _mean_only_bytes(artifact,cfg):
    D=artifact.mean.shape[1]; index_bytes=2 if D<=65535 else 4; value_bytes=max(1,int(np.ceil(cfg.quant_bits/8)))
    return int(artifact.mask.sum())*(index_bytes+value_bytes)+artifact.mean.shape[0]*4


def _evaluate_artifacts(label, artifacts, locals_, calibration_caches, eval_caches, oracle_summaries, oracle_counts, task_model, bundle, cfg, seed, device, logger):
    means=np.stack([a.mean for a in artifacts]); vars_=np.stack([a.var_mean for a in artifacts]); counts=np.stack([a.counts for a in artifacts]); masks=np.stack([a.mask for a in artifacts])
    n=len(artifacts); results={}

    # CFBA calibration is private and scenario-specific. For attacked priors we
    # recalibrate against the attacked global prior rather than reusing a clean certificate.
    needs_cfba=any(m in ('cfba_crc','global_crc') for m in cfg.alignment.methods)
    cfba_prior=xfedalign_prior(means) if needs_cfba else None
    certs=[]; caps=[]; global_beta=0.0
    if needs_cfba:
        grid=list(map(float,cfg.alignment.cfba_beta_grid)); alpha=float(cfg.alignment.cfba_alpha)
        for i in range(n):
            records=list(calibration_caches[i]['records'])[:int(cfg.alignment.cfba_calibration_samples_per_client)]
            if len(records) < int(cfg.alignment.cfba_min_calibration_samples):
                cert={'beta':0.0,'index':0,'certified':False,'upper_risk':[],'empirical_risk':[],'n_calibration':len(records),'reason':'insufficient_calibration'}
            else:
                L,_=calibration_loss_matrix(task_model,records,cfba_prior[i],grid,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k)
                cert=choose_largest_certified_beta(L,grid,alpha=alpha,bound=1.0)
            certs.append(cert); caps.append(float(cert['beta']))
        global_beta=float(min(caps)) if caps else 0.0
        logger.save_json(f'artifacts/{label}__cfba_certification.json',{
            'alpha':alpha,'beta_grid':grid,'per_client':certs,'global_safe_beta':global_beta,
            'certified_fraction':float(np.mean([bool(c['certified']) for c in certs])) if certs else 0.0,
        })

    for method in cfg.alignment.methods:
        priors,meta=make_priors(method,means,vars_,counts,masks,cfg.alignment,cfg.artifact,seed)
        if method=='cfba_crc': betas=np.asarray(caps,float)
        elif method=='global_crc': betas=np.full(n,global_beta,float)
        elif method=='local': betas=np.zeros(n,float)
        else: betas=np.full(n,_method_beta(method,cfg),float)
        summaries=np.stack([_aligned_summary(method,means[i],priors[i],float(betas[i])) for i in range(n)])
        p_edi=pairwise_edi(summaries,counts)
        if method=='local': ref=global_mean_prior(means,counts)
        else: ref=priors
        edi=reference_edi(summaries,ref,counts)
        fidelity=[]
        for i in range(n):
            valid=(counts[i]>0)&(oracle_counts[i]>0)
            v=summary_jsd(summaries[i],oracle_summaries[i],valid)
            if np.isfinite(v): fidelity.append(v)
        per_sample=[]
        for i in range(n):
            if method in ('local','xfedalign_median','cfba_crc','global_crc'):
                pm=evaluate_beta_on_records(task_model,eval_caches[i]['records'][:int(cfg.evaluation.deletion_insertion_samples_per_client)],priors[i],float(betas[i]),device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k)
                pm['n_eval_explanations']=pm.pop('n')
            else:
                pm=evaluate_method_cache(task_model,eval_caches[i],priors[i],method,float(betas[i]),cfg.evaluation,device)
            per_sample.append(pm)
        def avg(key):
            vals=[x[key] for x in per_sample if key in x and np.isfinite(x[key])]
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
            'excess_fidelity_risk':avg('excess_fidelity_risk'),
            'excess_fidelity_risk_p90':avg('excess_fidelity_risk_p90'),
            'communication_bytes_per_client_artifact':comm,
            'n_eval_explanations':int(sum(x.get('n_eval_explanations',0) for x in per_sample)),
            'selected_beta_mean':float(np.mean(betas)),
            'selected_beta_min':float(np.min(betas)),
            'selected_beta_max':float(np.max(betas)),
        }
        if method in ('cfba_crc','global_crc'):
            rec['certified_fraction']=float(np.mean([bool(c['certified']) for c in certs])) if certs else 0.0
        results[method]=rec
        for k,v in rec.items():
            if isinstance(v,(int,float)) and np.isfinite(v): logger.metric(k,v,stage='explanation',scenario=label,method=method)
        if method in ('cfba_crc','global_crc'):
            meta=dict(meta or {}); meta.update({'betas':betas.tolist(),'alpha':float(cfg.alignment.cfba_alpha),'certification':certs})
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
            'calibration_class_histograms':[_class_hist(x,bundle.n_classes) for x in bundle.calibration_clients],
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

        calibration_caches=[]
        for cid in range(cfg.federation.n_clients):
            cc=prepare_client_evaluation(task_model,bundle.calibration_clients[cid],locals_[cid].surrogate_state,cfg.surrogate.source,bundle.input_shape,bundle.n_classes,device,cfg.surrogate.ig_steps,seed+9000+cid*211,cfg.num_workers,max(int(cfg.alignment.cfba_calibration_samples_per_client),int(cfg.alignment.cfba_min_calibration_samples)))
            calibration_caches.append(cc)
        eval_caches=[];oracle=[];oc=[]
        for cid in range(cfg.federation.n_clients):
            cache=prepare_client_evaluation(task_model,bundle.eval_clients[cid],locals_[cid].surrogate_state,cfg.surrogate.source,bundle.input_shape,bundle.n_classes,device,cfg.surrogate.ig_steps,seed+cid*211,cfg.num_workers,cfg.evaluation.max_eval_samples)
            eval_caches.append(cache);oracle.append(cache['oracle_summary']);oc.append(cache['oracle_counts'])
        oracle=np.stack(oracle);oc=np.stack(oc)
        clean=_evaluate_artifacts('clean',artifacts,locals_,calibration_caches,eval_caches,oracle,oc,task_model,bundle,cfg,seed,device,logger)
        attacked=None;bad=[]
        if cfg.attack.enabled:
            attacked_art,bad=apply_artifact_attack(artifacts,cfg.attack,seed+8181)
            attacked=_evaluate_artifacts('attacked',attacked_art,locals_,calibration_caches,eval_caches,oracle,oc,task_model,bundle,cfg,seed,device,logger)
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
