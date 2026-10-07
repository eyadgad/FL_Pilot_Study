from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent/'cfba_core'
sys.path.insert(0,str(ROOT))

from ucpa_fl.config import ExperimentConfig, DatasetConfig, FederationConfig, ShiftConfig, SurrogateConfig, ArtifactConfig, AlignmentConfig, EvaluationConfig, AttackConfig, LoggingConfig
from ucpa_fl.datasets import load_data, make_loader
from ucpa_fl.federated import train_fedavg, evaluate
from ucpa_fl.repro import set_seed, resolve_device
from ucpa_fl.explain import build_local_explanation
from ucpa_fl.artifacts import sanitize_artifact
from ucpa_fl.alignment import xfedalign_prior, jsd
from ucpa_fl.metrics import prepare_client_evaluation, pairwise_edi, normalize_np, topk_overlap
from ucpa_fl.risk_control import perturbation_auc_many, crc_upper_empirical_risk, monotone_envelope_nondec

EPS=1e-12
BETA_GRID=np.array([0.0,0.1,0.2,0.4,0.6,0.8],float)
LAMBDA_GRID=np.array([0.0,0.25,0.5,0.75,1.0],float)
ALPHA=0.05


def make_cfg(shift,seed,outroot):
    return ExperimentConfig(
        experiment_name=f'multiprop_{shift}', seeds=[seed], device='auto', deterministic=True, num_workers=0,
        dataset=DatasetConfig(name='synthetic', train_limit=3200, test_limit=600, n_classes=4),
        federation=FederationConfig(n_clients=4, rounds=14, local_epochs=2, batch_size=64, lr=0.10, momentum=0.9, weight_decay=0.0, participation_rate=1.0, partition='dirichlet', dirichlet_alpha=0.8, min_client_samples=450),
        shift=ShiftConfig(kind=shift, rotation_max_deg=32.0, patch_size=4, patch_value=1.0),
        surrogate=SurrogateConfig(kind='linear', source='linear_surrogate', epochs=3, lr=0.05, temperature=2.0, l1=1e-4, hidden_dim=32, max_samples_per_class=24, eval_samples_per_class=12, ig_steps=8, artifact_samples_per_class=16),
        artifact=ArtifactConfig(topk=64, clip_radius=5.0, quant_bits=8, dp_sigma=0.05, missing_variance_scale=4.0, send_variance=False, sparse_intersection_only=True),
        alignment=AlignmentConfig(beta=0.2, methods=['local','xfedalign_median']),
        evaluation=EvaluationConfig(deletion_steps=3, eval_batch_size=64, max_eval_samples=60, topk_overlap_k=64, deletion_insertion_samples_per_client=12),
        attack=AttackConfig(enabled=False), logging=LoggingConfig(output_root=str(outroot), save_checkpoints=False, save_artifacts=False, log_every_round=False),
        tags={'purpose':'multi_proposal_adversarial_smoke'})


def split3(records,seed,n_select=12,n_cal=32,n_test=8):
    rec=list(records); rng=np.random.default_rng(int(seed)); rec=[rec[i] for i in rng.permutation(len(rec))]
    n=len(rec)
    if n < n_select+n_cal+n_test:
        n_select=max(12,n//3); n_cal=max(12,(n-n_select)//2); n_test=n-n_select-n_cal
    return rec[:n_select],rec[n_select:n_select+n_cal],rec[n_select+n_cal:n_select+n_cal+n_test]


def basic_maps(rec,prior,beta):
    l=normalize_np(np.asarray(rec['local_map'])[None])[0]; p=normalize_np(np.asarray(prior)[None])[0]
    return normalize_np(((1-beta)*l+beta*p)[None])[0]


def weighted_map(rec,prior,lam,kind,param=1.0):
    l=normalize_np(np.asarray(rec['local_map'])[None])[0]; p=normalize_np(np.asarray(prior)[None])[0]
    D=len(l); order=np.argsort(-l); rank=np.empty(D,float); rank[order]=np.linspace(0,1,D) # 0 most important,1 least
    if kind=='importance':
        w=np.power(rank,float(param))
    elif kind=='hybrid':
        # align low-importance coords only when local/global disagreement is modest
        scale=np.median(np.abs(l-p))+1e-8
        w=np.power(rank,float(param))*np.exp(-np.abs(l-p)/(2*scale))
    else: raise ValueError(kind)
    a=np.clip(lam*w,0,1)
    return normalize_np(((1-a)*l+a*p)[None])[0]


def gated_map(rec,prior,beta,threshold):
    l=normalize_np(np.asarray(rec['local_map'])[None])[0]; p=normalize_np(np.asarray(prior)[None])[0]
    if float(jsd(l,p)) <= float(threshold):
        return normalize_np(((1-beta)*l+beta*p)[None])[0]
    return l


def class_map(rec,prior,class_betas,scale):
    c=int(rec['pred']); return basic_maps(rec,prior,float(class_betas[c])*float(scale))


def uncertainty_map(rec,prior,lam,var_summary,z=1.5):
    # private uncertainty compatibility: align coordinates where class-summary deviation is within z*SE;
    # sample local map supplies final base. A soft sigmoid makes the policy stable.
    c=int(rec['pred']); l=normalize_np(np.asarray(rec['local_map'])[None])[0]; p=normalize_np(np.asarray(prior)[None])[0]
    mu=normalize_np(np.asarray(rec.get('class_mean',l))[None])[0] if 'class_mean' in rec else l
    v=np.asarray(var_summary[c],float); scale=np.sqrt(np.maximum(v,0)+1e-8)
    compat=np.exp(-0.5*((mu-p)/(z*scale+1e-8))**2)
    a=np.clip(float(lam)*compat,0,1)
    return normalize_np(((1-a)*l+a*p)[None])[0]


def cheap_metric(records,maps,topk):
    js=[];tops=[];harms=[];move=[]
    for rec,m in zip(records,maps):
        l=normalize_np(np.asarray(rec['local_map'])[None])[0]; o=normalize_np(np.asarray(rec['oracle_map'])[None])[0]
        k0=topk_overlap(l,o,min(topk,len(l))); k1=topk_overlap(m,o,min(topk,len(l)))
        j0=float(jsd(l,o)); j1=float(jsd(m,o))
        h=max(0.0,float(k0-k1),float(j1-j0))
        js.append(j1);tops.append(float(k1));harms.append(h)
        p=normalize_np(np.asarray(rec['prior'])[None])[0] if 'prior' in rec else None
        move.append(float(jsd(l,p)-jsd(m,p)) if p is not None else 0.0)
    return dict(sample_fidelity_jsd=float(np.mean(js)),topk_oracle_overlap=float(np.mean(tops)),risk=float(np.mean(harms)),risk_p90=float(np.quantile(harms,.9)),movement=float(np.mean(move)))

def metric_for_maps(model,records,maps,device,steps,topk):
    js=[];dele=[];ins=[];tops=[];harms=[];move=[]
    for rec,m in zip(records,maps):
        l=normalize_np(np.asarray(rec['local_map'])[None])[0]; o=normalize_np(np.asarray(rec['oracle_map'])[None])[0]
        c=int(rec['pred']); x=torch.as_tensor(rec['x'],dtype=torch.float32)
        dm,im=perturbation_auc_many(model,x,c,np.stack([l,m]),device,steps)
        k0=topk_overlap(l,o,min(topk,len(l))); k1=topk_overlap(m,o,min(topk,len(l)))
        j0=float(jsd(l,o)); j1=float(jsd(m,o))
        h=max(0.0,float(dm[1]-dm[0]),float(im[0]-im[1]),float(k0-k1),float(j1-j0))
        js.append(j1);dele.append(float(dm[1]));ins.append(float(im[1]));tops.append(float(k1));harms.append(h)
        p=normalize_np(np.asarray(rec['prior'])[None])[0] if 'prior' in rec else None
        move.append(float(jsd(l,p)-jsd(m,p)) if p is not None else 0.0)
    return dict(sample_fidelity_jsd=float(np.mean(js)),deletion_auc=float(np.mean(dele)),insertion_auc=float(np.mean(ins)),topk_oracle_overlap=float(np.mean(tops)),risk=float(np.mean(harms)),risk_p90=float(np.quantile(harms,.9)),movement=float(np.mean(move)))


def losses_for_policy(model,records,mapper_grid,device,steps,topk):
    # Cheap certification smoke: excess JSD/top-k harm only. Full perturbation metrics are held for test.
    L=[]
    for rec in records:
        l=normalize_np(np.asarray(rec['local_map'])[None])[0]; o=normalize_np(np.asarray(rec['oracle_map'])[None])[0]
        k0=topk_overlap(l,o,min(topk,len(l))); j0=float(jsd(l,o))
        row=[]
        for fn in mapper_grid:
            m=fn(rec); k=topk_overlap(m,o,min(topk,len(l))); j=float(jsd(m,o))
            row.append(max(0.0,float(k0-k),float(j-j0)))
        L.append(row)
    return np.asarray(L,float)

def choose_crc(L,grid,alpha=ALPHA):
    env=monotone_envelope_nondec(L); upper=crc_upper_empirical_risk(env,1.0); ok=np.where(upper<=alpha+1e-12)[0]
    idx=int(ok[-1]) if len(ok) else 0
    return float(grid[idx]),int(idx),upper.tolist(),env.mean(0).tolist(),bool(len(ok))


def tune_sample_gate(model,sel,prior,device,steps,topk):
    # use independent policy-selection data, so this data may choose threshold/beta freely.
    dists=np.array([float(jsd(normalize_np(np.asarray(r['local_map'])[None])[0],prior[int(r['pred'])])) for r in sel])
    thresholds=np.unique(np.quantile(dists,[.25,.5,.75,1.0]))
    best=None
    for beta in [0.2,0.4,0.6]:
      for t in thresholds:
        maps=[gated_map(r,prior[int(r['pred'])],beta,t) for r in sel]
        for r in sel: r['prior']=prior[int(r['pred'])]
        m=cheap_metric(sel,maps,topk)
        # selection objective: maximize movement, constrain empirical harm <=.04
        if m['risk']<=.04:
            score=m['movement']
            if best is None or score>best[0]:best=(score,beta,float(t),m)
    if best is None: return 0.0,float(np.min(dists)),{}
    return best[1],best[2],best[3]


def tune_class_shape(model,sel,prior,C,device,steps,topk):
    b=np.zeros(C,float)
    for c in range(C):
        rc=[r for r in sel if int(r['pred'])==c]
        if len(rc)<3: continue
        best=(0.0,0.0)
        for beta in [0.1,0.2,0.4,0.6,0.8]:
            maps=[basic_maps(r,prior[c],beta) for r in rc]
            for r in rc:r['prior']=prior[c]
            m=cheap_metric(rc,maps,topk)
            if m['risk']<=.05 and m['movement']>best[0]:best=(m['movement'],beta)
        b[c]=best[1]
    return b


def tune_weight_shape(model,sel,prior,kind,device,steps,topk):
    best=None
    for param in ([0.5,1.0,2.0] if kind=='importance' else [0.5,1.0,2.0]):
        maps=[weighted_map(r,prior[int(r['pred'])],1.0,kind,param) for r in sel]
        for r in sel:r['prior']=prior[int(r['pred'])]
        m=cheap_metric(sel,maps,topk)
        # choose shape by best movement/risk tradeoff, no requirement of full-scale lambda safety
        score=m['movement']-3*m['risk']
        if best is None or score>best[0]:best=(score,param,m)
    return float(best[1]),best[2]


def summarize_edi(records_by_client,priors,policy_fns):
    summaries=[];counts=[]
    C,D=priors.shape[1],priors.shape[2]
    for recs,fn in zip(records_by_client,policy_fns):
        buckets=[[] for _ in range(C)]
        for r in recs:buckets[int(r['pred'])].append(fn(r))
        s=np.zeros((C,D));ct=np.zeros(C,int)
        for c,v in enumerate(buckets):
            if v:s[c]=normalize_np(np.mean(v,0)[None])[0];ct[c]=len(v)
        summaries.append(s);counts.append(ct)
    return float(pairwise_edi(np.stack(summaries),np.stack(counts)))


def run(seed,shift,outroot):
    cfg=make_cfg(shift,seed,outroot);set_seed(seed,True);device=resolve_device('auto');bundle=load_data(cfg,seed)
    class DummyLogger:
      def log(self,*a,**k):pass
      def metric(self,*a,**k):pass
      def save_json(self,*a,**k):pass
    model,task=train_fedavg(cfg,bundle,seed,device,DummyLogger())
    acc=[]
    for cid,ds in enumerate(bundle.eval_clients):acc.append(evaluate(model,make_loader(ds,128,False,seed+6000+cid,0),device)['accuracy'])
    bundle.set_round(cfg.federation.rounds-1);locals_=[];arts=[];rng=np.random.default_rng(seed+70707)
    for cid in range(cfg.federation.n_clients):
        le=build_local_explanation(model,bundle.surrogate_clients[cid],bundle.artifact_clients[cid],bundle.input_shape,bundle.n_classes,cfg.surrogate,seed+cid*97,device,0)
        locals_.append(le);arts.append(sanitize_artifact(le.mean,le.var_mean,le.counts,cfg.artifact,rng))
    means=np.stack([a.mean for a in arts]);prior_all=xfedalign_prior(means);var_all=np.stack([le.var_mean for le in locals_])
    sels=[];cals=[];tests=[]
    for cid in range(cfg.federation.n_clients):
        cache=prepare_client_evaluation(model,bundle.eval_clients[cid],locals_[cid].surrogate_state,cfg.surrogate.source,bundle.input_shape,bundle.n_classes,device,cfg.surrogate.ig_steps,seed+cid*211,0,cfg.evaluation.max_eval_samples)
        se,ca,te=split3(cache['records'],seed+cid*101);sels.append(se);cals.append(ca);tests.append(te)
    proposal_fns={}; diagnostics={}
    # baselines
    proposal_fns['local']=[lambda r: normalize_np(np.asarray(r['local_map'])[None])[0] for _ in range(4)]
    proposal_fns['xfedalign_0p2']=[(lambda i: (lambda r: basic_maps(r,prior_all[i,int(r['pred'])],0.2)))(i) for i in range(4)]
    # P1 sample gate: select beta/t on selection, certify scale gamma in [0,1] multiplying beta
    fns=[];diag=[]
    for i in range(4):
        beta,t,dm=tune_sample_gate(model,sels[i],prior_all[i],device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k)
        mappers=[(lambda lam: (lambda r: gated_map(r,prior_all[i,int(r['pred'])],beta*lam,t)))(lam) for lam in LAMBDA_GRID]
        L=losses_for_policy(model,cals[i],mappers,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k)
        lam,idx,up,emp,cert=choose_crc(L,LAMBDA_GRID)
        fns.append((lambda i=i,beta=beta,t=t,lam=lam: (lambda r: gated_map(r,prior_all[i,int(r['pred'])],beta*lam,t)))())
        diag.append({'beta_shape':beta,'threshold':t,'lambda':lam,'crc_upper':up,'certified':cert})
    proposal_fns['sample_gate_crc']=fns;diagnostics['sample_gate_crc']=diag
    # P2 class policy shape + certified global scaling per client
    fns=[];diag=[]
    for i in range(4):
        b=tune_class_shape(model,sels[i],prior_all[i],bundle.n_classes,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k)
        mappers=[(lambda lam: (lambda r: class_map(r,prior_all[i,int(r['pred'])],b,lam)))(lam) for lam in LAMBDA_GRID]
        L=losses_for_policy(model,cals[i],mappers,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k)
        lam,idx,up,emp,cert=choose_crc(L,LAMBDA_GRID)
        fns.append((lambda i=i,b=b.copy(),lam=lam: (lambda r: class_map(r,prior_all[i,int(r['pred'])],b,lam)))())
        diag.append({'class_beta_shape':b.tolist(),'lambda':lam,'crc_upper':up,'certified':cert})
    proposal_fns['class_policy_crc']=fns;diagnostics['class_policy_crc']=diag
    # P3 importance weighted shape + CRC scale
    for name,kind in [('importance_crc','importance'),('hybrid_crc','hybrid')]:
      fns=[];diag=[]
      for i in range(4):
        param,dm=tune_weight_shape(model,sels[i],prior_all[i],kind,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k)
        mappers=[(lambda lam: (lambda r: weighted_map(r,prior_all[i,int(r['pred'])],lam,kind,param)))(lam) for lam in LAMBDA_GRID]
        L=losses_for_policy(model,cals[i],mappers,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k)
        lam,idx,up,emp,cert=choose_crc(L,LAMBDA_GRID)
        fns.append((lambda i=i,param=param,lam=lam,kind=kind: (lambda r: weighted_map(r,prior_all[i,int(r['pred'])],lam,kind,param)))())
        diag.append({'shape_param':param,'lambda':lam,'crc_upper':up,'certified':cert})
      proposal_fns[name]=fns;diagnostics[name]=diag
    # P5 uncertainty compatible summary mask; choose z on selection, then CRC lambda
    fns=[];diag=[]
    for i in range(4):
      # var summary local private. test several z shapes on selection
      best=None
      for z in [0.75,1.5,3.0]:
        maps=[]
        for r in sels[i]:
          # inject class mean to calculate compatibility at summary level
          rr=dict(r); rr['class_mean']=locals_[i].mean[int(r['pred'])]
          maps.append(uncertainty_map(rr,prior_all[i,int(r['pred'])],1.0,var_all[i],z))
          rr['prior']=prior_all[i,int(r['pred'])]
        # use original recs for metrics; maps sufficient
        for r in sels[i]:r['prior']=prior_all[i,int(r['pred'])]
        m=metric_for_maps(model,sels[i],maps,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k)
        score=m['movement']-3*m['risk']
        if best is None or score>best[0]:best=(score,z,m)
      z=float(best[1])
      def mk(lam,z=z,i=i):
        def fn(r):
          rr=dict(r);rr['class_mean']=locals_[i].mean[int(r['pred'])]
          return uncertainty_map(rr,prior_all[i,int(r['pred'])],lam,var_all[i],z)
        return fn
      mappers=[mk(lam) for lam in LAMBDA_GRID]
      L=losses_for_policy(model,cals[i],mappers,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k)
      lam,idx,up,emp,cert=choose_crc(L,LAMBDA_GRID)
      fns.append(mk(lam));diag.append({'z':z,'lambda':lam,'crc_upper':up,'certified':cert})
    proposal_fns['uncertainty_crc']=fns;diagnostics['uncertainty_crc']=diag

    metrics={}
    for name,fns in proposal_fns.items():
        per=[]
        for i in range(4):
            maps=[fns[i](r) for r in tests[i]]
            for r in tests[i]:r['prior']=prior_all[i,int(r['pred'])]
            per.append(metric_for_maps(model,tests[i],maps,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k))
        avg=lambda k:float(np.mean([p[k] for p in per]))
        metrics[name]={k:avg(k) for k in ['sample_fidelity_jsd','deletion_auc','insertion_auc','topk_oracle_overlap','risk','risk_p90','movement']}
        metrics[name]['pairwise_edi']=summarize_edi(tests,prior_all,fns)
    return {'seed':seed,'shift':shift,'task_acc_mean':float(np.mean(acc)),'task_acc_min':float(np.min(acc)),'metrics':metrics,'diagnostics':diagnostics}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='/mnt/data/multiproposal_smoke/results');ap.add_argument('--seeds',default='1801,1802');ap.add_argument('--shifts',default='rotation,patch,erasing');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);R=[]
    for sh in args.shifts.split(','):
      for s in map(int,args.seeds.split(',')):
        print('RUN',sh,s,flush=True);r=run(s,sh,out);R.append(r);(out/f'{sh}_{s}.json').write_text(json.dumps(r,indent=2));print('DONE',sh,s,r['task_acc_mean'],flush=True)
    (out/'all.json').write_text(json.dumps(R,indent=2))
