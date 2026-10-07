from __future__ import annotations
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np
import torch

from ucpa_fl.config import ExperimentConfig, DatasetConfig, FederationConfig, ShiftConfig, SurrogateConfig, ArtifactConfig, AlignmentConfig, EvaluationConfig, AttackConfig, LoggingConfig
from ucpa_fl.datasets import load_data
from ucpa_fl.federated import train_fedavg, evaluate
from ucpa_fl.logging_utils import RunLogger
from ucpa_fl.repro import set_seed, resolve_device, environment_snapshot
from ucpa_fl.explain import build_local_explanation
from ucpa_fl.artifacts import sanitize_artifact
from ucpa_fl.alignment import xfedalign_prior
from ucpa_fl.metrics import prepare_client_evaluation, pairwise_edi, normalize_np
from ucpa_fl.risk_control import calibration_loss_matrix, choose_largest_certified_beta, evaluate_beta_on_records, aligned_map, optimize_betas_under_caps
from ucpa_fl.datasets import make_loader


def make_cfg(shift, seed, outroot):
    return ExperimentConfig(
        experiment_name=f'cfba_crc_{shift}', seeds=[seed], device='cpu', deterministic=True, num_workers=0,
        dataset=DatasetConfig(name='synthetic', train_limit=3200, test_limit=500, n_classes=4),
        federation=FederationConfig(n_clients=4, rounds=14, local_epochs=2, batch_size=64, lr=0.10, momentum=0.9, weight_decay=0.0, participation_rate=1.0, partition='dirichlet', dirichlet_alpha=0.8, min_client_samples=450),
        shift=ShiftConfig(kind=shift, rotation_max_deg=32.0, patch_size=4, patch_value=1.0),
        surrogate=SurrogateConfig(kind='linear', source='linear_surrogate', epochs=3, lr=0.05, temperature=2.0, l1=1e-4, hidden_dim=32, max_samples_per_class=24, eval_samples_per_class=12, ig_steps=8, artifact_samples_per_class=16),
        artifact=ArtifactConfig(topk=64, clip_radius=5.0, quant_bits=8, dp_sigma=0.05, missing_variance_scale=4.0, send_variance=False, sparse_intersection_only=True),
        alignment=AlignmentConfig(beta=0.2, methods=['local','xfedalign_median']),
        evaluation=EvaluationConfig(deletion_steps=4, eval_batch_size=64, max_eval_samples=72, topk_overlap_k=64, deletion_insertion_samples_per_client=24),
        attack=AttackConfig(enabled=False),
        logging=LoggingConfig(output_root=str(outroot), save_checkpoints=False, save_artifacts=True, log_every_round=False),
        tags={'purpose':'cfba_crc_adversarial_smoke'}
    )


def split_cache(cache, seed, n_cal=32, n_test=16):
    rec=list(cache['records'])
    rng=np.random.default_rng(int(seed)); order=rng.permutation(len(rec)); rec=[rec[i] for i in order]
    if len(rec) < n_cal+n_test:
        n_cal=max(12,len(rec)//2); n_test=len(rec)-n_cal
    return rec[:n_cal], rec[n_cal:n_cal+n_test]


def summarize_alignment(records_by_client, priors, betas):
    summaries=[]; counts=[]
    for recs,prior,b in zip(records_by_client,priors,betas):
        # summary over the actual final per-sample aligned explanations, grouped by predicted class
        C=prior.shape[0]; D=prior.shape[1]; buckets=[[] for _ in range(C)]
        for r in recs:
            c=int(r['pred']); buckets[c].append(aligned_map(r['local_map'],prior[c],float(b)))
        s=np.zeros((C,D)); ct=np.zeros(C,int)
        for c,vals in enumerate(buckets):
            if vals: s[c]=normalize_np(np.mean(vals,0)[None])[0];ct[c]=len(vals)
        summaries.append(s);counts.append(ct)
    return np.stack(summaries),np.stack(counts)


def run(seed, shift, outroot, alpha=0.05):
    cfg=make_cfg(shift,seed,outroot); set_seed(seed,True); device=resolve_device(cfg.device)
    logger=RunLogger(str(outroot), cfg.experiment_name, seed, cfg.to_dict(), environment_snapshot())
    bundle=load_data(cfg,seed)
    model,task=train_fedavg(cfg,bundle,seed,device,logger)
    # deployment client task accuracy
    acc=[]
    for cid,ds in enumerate(bundle.eval_clients):
        acc.append(evaluate(model,make_loader(ds,128,False,seed+6000+cid,0),device)['accuracy'])
    bundle.set_round(cfg.federation.rounds-1)
    locals_=[];arts=[];rng=np.random.default_rng(seed+70707)
    for cid in range(cfg.federation.n_clients):
        le=build_local_explanation(model,bundle.surrogate_clients[cid],bundle.artifact_clients[cid],bundle.input_shape,bundle.n_classes,cfg.surrogate,seed+cid*97,device,0)
        locals_.append(le);arts.append(sanitize_artifact(le.mean,le.var_mean,le.counts,cfg.artifact,rng))
    means=np.stack([a.mean for a in arts]); priors=xfedalign_prior(means)
    cal_recs=[];test_recs=[]
    for cid in range(cfg.federation.n_clients):
        cache=prepare_client_evaluation(model,bundle.eval_clients[cid],locals_[cid].surrogate_state,cfg.surrogate.source,bundle.input_shape,bundle.n_classes,device,cfg.surrogate.ig_steps,seed+cid*211,0,cfg.evaluation.max_eval_samples)
        ca,te=split_cache(cache,seed+cid*101,n_cal=32,n_test=16); cal_recs.append(ca);test_recs.append(te)
    grid=[0.0,0.1,0.2,0.4,0.6,0.8]
    cert=[];loss_mats=[]
    for i in range(cfg.federation.n_clients):
        L,_=calibration_loss_matrix(model,cal_recs[i],priors[i],grid,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k)
        loss_mats.append(L); cert.append(choose_largest_certified_beta(L,grid,alpha=alpha,bound=1.0))
    client_beta=[x['beta'] for x in cert]
    global_beta=min(client_beta) # one beta certified for every client
    counts=np.stack([a.counts for a in arts])
    opt_beta,opt_artifact_edi=optimize_betas_under_caps(means,counts,priors,grid,client_beta,pairwise_edi)
    fixed_beta=0.2
    methods={'local':[0.0]*cfg.federation.n_clients,'global_crc':[global_beta]*cfg.federation.n_clients,'client_max_crc':client_beta,'cfba_crc':opt_beta,'xfedalign':[fixed_beta]*cfg.federation.n_clients,'fixed_0p4':[0.4]*cfg.federation.n_clients,'fixed_0p6':[0.6]*cfg.federation.n_clients}
    metrics={}; per_client={}
    for name,betas in methods.items():
        per=[evaluate_beta_on_records(model,test_recs[i],priors[i],betas[i],device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k) for i in range(cfg.federation.n_clients)]
        per_client[name]=per
        def avg(k): return float(np.mean([x[k] for x in per if np.isfinite(x[k])]))
        summaries,counts=summarize_alignment(test_recs,priors,betas)
        metrics[name]={k:avg(k) for k in ('sample_fidelity_jsd','deletion_auc','insertion_auc','topk_oracle_overlap','excess_fidelity_risk','excess_fidelity_risk_p90')}
        metrics[name]['pairwise_edi']=float(pairwise_edi(summaries,counts))
    result={'seed':seed,'shift':shift,'task_accuracy_mean':float(np.mean(acc)),'task_accuracy_min':float(np.min(acc)),'alpha':alpha,'betas':{'caps':client_beta,'global':global_beta,'cfba':opt_beta,'xfedalign':fixed_beta},'artifact_objective_edi':opt_artifact_edi,'certification':cert,'metrics':metrics,'per_client_metrics':per_client}
    p=Path(logger.run_dir)/'cfba_smoke_result.json';p.write_text(json.dumps(result,indent=2))
    logger.save_json('final_results.json',result);logger.finalize('completed')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='./cfba_smoke_outputs');ap.add_argument('--seeds',default='1601,1602,1603,1604,1605');ap.add_argument('--shifts',default='rotation,patch,erasing');ap.add_argument('--alpha',type=float,default=0.05);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    allr=[]
    for shift in args.shifts.split(','):
        for s in [int(x) for x in args.seeds.split(',') if x]:
            print('RUN',shift,s,flush=True);allr.append(run(s,shift,out,args.alpha))
    (out/'cfba_smoke_all.json').write_text(json.dumps(allr,indent=2))

if __name__=='__main__':main()
