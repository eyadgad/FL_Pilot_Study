#!/usr/bin/env python3
"""General real-MNIST seed-level comparison of frozen FAGC vs fair controls.

Fail closed on missing real MNIST or low CNN accuracy. Does not train/checkpoint
CNNs itself; separate independent FedAvg training manifests are required.
"""
from __future__ import annotations
import sys,json,hashlib,csv,argparse,time
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ucpa_fl.config import load_config
from ucpa_fl.repro import set_seed
from ucpa_fl.datasets import load_data,make_loader
from ucpa_fl.models import MNISTCNN
from ucpa_fl.federated import evaluate
from ucpa_fl.explain import build_local_explanation,explanation_batch
from ucpa_fl.artifacts import sanitize_artifact
from ucpa_fl.metrics import prepare_client_evaluation,normalize_np,pairwise_edi,topk_overlap
from ucpa_fl.alignment import xfedalign_prior,jsd
from ucpa_fl.residual import corrected_map
from ucpa_fl.successors import make_successors
from ucpa_fl.fagc import train_fagc,fagc_map
from ucpa_fl.risk_control import perturbation_auc_many
from scripts.verify_real_mnist import verify as verify_idx

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(kind,seed,replica,n_eval,root_out,config_path=None,checkpoint_override=None,training_rounds_override=None,n_teacher=12):
    freeze=json.loads((ROOT/'configs/fagc_frozen_confirmation.json').read_text())
    if checkpoint_override is None and seed not in freeze['trained_task_seeds'][kind]:raise ValueError('seed not in frozen confirmation matrix')
    cfg=load_config(config_path or ROOT/'configs'/f'mnist_real_qualified_{kind}.yaml')
    cfg.dataset.root=str(ROOT/'data');cfg.dataset.test_limit=0
    if cfg.dataset.name!='mnist' or cfg.dataset.download:raise ValueError('must use genuine MNIST offline')
    set_seed(seed+100001*replica,True)
    bundle=load_data(cfg,seed)
    split_index_lists={str(i):{'task':bundle.task_clients[i].indices.tolist(),'surrogate':bundle.surrogate_clients[i].indices.tolist(),
      'artifact':bundle.artifact_clients[i].indices.tolist(),'teacher':bundle.calibration_clients[i].indices.tolist(),
      'evaluation':bundle.eval_clients[i].indices.tolist()} for i in range(cfg.federation.n_clients)}
    partition_hash=hashlib.sha256(json.dumps(split_index_lists,sort_keys=True).encode()).hexdigest()
    if checkpoint_override is not None:
      checkpoint=Path(checkpoint_override);training_rounds=int(training_rounds_override or cfg.federation.rounds)
      if not checkpoint.is_file():raise RuntimeError('missing checkpoint in general FL suite')
    elif seed<6000:
      ckpts=list((ROOT/'real_mnist_smoke_outputs').glob(f'real_mnist_{kind}_qualified_smoke/*/checkpoints/global_task_model.pt'))
      if len(ckpts)!=1:raise RuntimeError('ambiguous original checkpoint')
      checkpoint=ckpts[0];training_rounds=12
    else:
      out=ROOT/'fagc_independent_checkpoints'/kind/str(seed)
      meta=json.loads((out/'manifest.json').read_text());checkpoint=out/'global_task_model.pt';training_rounds=meta['rounds']
      if meta['seed']!=seed or meta['kind']!=kind or sha(checkpoint)!=meta['checkpoint_sha256']:raise ValueError('independent training manifest/checkpoint mismatch')
      # Hash partition indices before using split and compare with the independently trained saved configuration.
      partition={str(i):{'task':bundle.task_clients[i].indices.tolist(),'surrogate':bundle.surrogate_clients[i].indices.tolist(),'artifact':bundle.artifact_clients[i].indices.tolist(),'teacher':bundle.calibration_clients[i].indices.tolist(),'evaluation':bundle.eval_clients[i].indices.tolist()} for i in range(cfg.federation.n_clients)}
      p_hash=hashlib.sha256(json.dumps(partition,sort_keys=True).encode()).hexdigest()
      if p_hash!=meta['partition_sha256']:raise ValueError('task-vs-explanation partition mismatch')
    model=MNISTCNN().cpu();model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'));model.eval()
    checkpoint_hash=sha(checkpoint)
    acc=evaluate(model,make_loader(bundle.test,256,False,seed,0),torch.device('cpu'))
    if len(bundle.test)!=10000 or acc['accuracy']<freeze['task_accuracy_min_full10k']:
      print('FAILED TASK GATE',kind,seed,acc,flush=True);return {'failed_task_accuracy':acc}
    print('TASK GATE PASSED',kind,seed,acc['accuracy'],flush=True)
    bundle.set_round(training_rounds-1)
    device=torch.device('cpu');rng=np.random.default_rng(seed+replica*100001+70707)
    locals_,artifacts=[],[];art_recs=[]
    for i in range(cfg.federation.n_clients):
        le=build_local_explanation(model,bundle.surrogate_clients[i],bundle.artifact_clients[i],bundle.input_shape,bundle.n_classes,cfg.surrogate,seed+i*97+replica*100001,device,0)
        locals_.append(le);artifacts.append(sanitize_artifact(le.mean,le.var_mean,le.counts,cfg.artifact,rng))
        # Exact artifact EDI: average the same maps actually served on real training-side images.
        # No teacher or evaluation records enter this step.
        bucket=[[] for _ in range(10)];maxper=cfg.surrogate.artifact_samples_per_class
        for x,_ in make_loader(bundle.artifact_clients[i],64,False,seed+i*97+replica*100001+17,0):
            loc,pred=explanation_batch(model,le.surrogate_state,cfg.surrogate.source,x,bundle.input_shape,bundle.n_classes,device,cfg.surrogate.ig_steps)
            loc=loc.numpy();pred=pred.numpy();x=x.numpy()
            for j,c in enumerate(pred):
                if len(bucket[c])<maxper:bucket[c].append({'x':x[j],'pred':int(c),'local_map':loc[j]})
            if all(len(row)>=maxper for row in bucket):break
        art_recs.append(bucket)
        if [len(row) for row in bucket]!=[int(t) for t in le.counts]:raise RuntimeError('real-artifact sample count drift')
        print('surrogate/artifact done',kind,seed,i,flush=True)
    teachers=[];evals=[]
    for i in range(cfg.federation.n_clients):
        teachers.append(prepare_client_evaluation(model,bundle.calibration_clients[i],locals_[i].surrogate_state,cfg.surrogate.source,bundle.input_shape,bundle.n_classes,device,cfg.surrogate.ig_steps,seed+9000+i*211,0,n_teacher)['records'])
        evals.append(prepare_client_evaluation(model,bundle.eval_clients[i],locals_[i].surrogate_state,cfg.surrogate.source,bundle.input_shape,bundle.n_classes,device,cfg.surrogate.ig_steps,seed+i*211,0,n_eval)['records'])
        if len(teachers[-1])!=n_teacher or len(evals[-1])!=n_eval:raise RuntimeError('insufficient disjoint real IG examples')
        print('teacher and heldout IG ready',kind,seed,i,flush=True)
    prior=xfedalign_prior(np.stack([a.mean for a in artifacts]))
    residual,oldcodec=make_successors(teachers,10,784,48,kind,cfg.shift.rotation_max_deg)
    opts=freeze['fagc_parameters']
    fields,ledger=train_fagc(teachers,power=opts['input_power'],sigma=opts['teacher_blur_sigma_px'],peer_fraction=opts['peer_fraction'])
    privfields=np.stack([train_fagc([t],power=opts['input_power'],sigma=opts['teacher_blur_sigma_px'],peer_fraction=0)[0][0] for t in teachers])
    methods=['local','xfedalign_style','private_beta_0','private_beta_04','input_only','private_field','fagc_no_class','fagc']
    def get_map(m,i,c,l,x):
        l=normalize_np(np.asarray(l).reshape(1,-1))[0]
        if m=='local':return l
        if m=='xfedalign_style':return normalize_np((.8*l+.2*prior[i,c])[None])[0]
        if m=='private_beta_0':return corrected_map(l,prior[i,c],residual['private'][i,c],.6,0.)
        if m=='private_beta_04':return corrected_map(l,prior[i,c],residual['private'][i,c],.6,.4)
        if m=='input_only':return normalize_np(np.asarray(x).reshape(1,-1).astype(float)**opts['input_power'])[0]
        if m=='private_field':return fagc_map(x,l,privfields[i],power=opts['input_power'],class_exponent=opts['class_exponent'])
        if m=='fagc_no_class':return fagc_map(x,l,fields[i],power=opts['input_power'],class_exponent=0.)
        if m=='fagc':return fagc_map(x,l,fields[i],power=opts['input_power'],class_exponent=opts['class_exponent'])
        raise ValueError(m)
    records=[]
    for i,recs in enumerate(evals):
        for j,r in enumerate(recs):
            c=int(r['pred']);o=normalize_np(np.asarray(r['oracle_map'])[None])[0]
            vv=[get_map(m,i,c,r['local_map'],r['x']) for m in methods]
            dels,ins=perturbation_auc_many(model,torch.as_tensor(r['x'],dtype=torch.float32),c,np.stack(vv),device,steps=3)
            row={'kind':kind,'task_seed':seed,'replica':replica,'client':i,'index':j,'pred_class':c}
            for k,m in enumerate(methods):
                row[m+'_jsd']=float(jsd(vv[k],o));row[m+'_topk']=float(topk_overlap(vv[k],o,48))
                row[m+'_deletion_auc']=float(dels[k]);row[m+'_insertion_auc']=float(ins[k])
            records.append(row)
    artmaps={m:np.zeros((cfg.federation.n_clients,10,784)) for m in methods};vcounts=np.zeros((cfg.federation.n_clients,10),int)
    for i in range(cfg.federation.n_clients):
        for c in range(10):
            recs=art_recs[i][c];vcounts[i,c]=len(recs)
            if not recs:continue
            for m in methods:
                artmaps[m][i,c]=normalize_np(np.mean([get_map(m,i,c,r['local_map'],r['x']) for r in recs],axis=0)[None])[0]
    result={}
    for m in methods:
        result[m]={'jsd':float(np.mean([r[m+'_jsd'] for r in records])),
                   'edi':float(pairwise_edi(artmaps[m],vcounts)),
                   'topk':float(np.mean([r[m+'_topk'] for r in records])),
                   'deletion_auc':float(np.mean([r[m+'_deletion_auc'] for r in records])),
                   'insertion_auc':float(np.mean([r[m+'_insertion_auc'] for r in records])),
                   'bytes_per_client':float(1960+np.mean(ledger['uplink_bytes'])+np.mean(ledger['downlink_bytes']) if m.startswith('fagc') else 1960)}
    if sha(checkpoint)!=checkpoint_hash:raise RuntimeError('frozen task checkpoint changed')
    output=Path(root_out)/f'{kind}_seed{seed}_replica{replica}';output.mkdir(parents=True,exist_ok=True)
    summary={'dataset':'Verified real-MNIST 60000 train / 10000 test CSV-origin IDX','seed':seed,'kind':kind,'replica':replica,'full_test_acc':acc['accuracy'],
        'full_test_n':10000,'checkpoint_sha256':checkpoint_hash,'n_teacher_per_client':list(map(len,teachers)),
        'n_eval_per_client':list(map(len,evals)),'teacher_test_disjoint':True,
        'train_rounds':training_rounds,'n_clients':cfg.federation.n_clients,'configured_teacher_count':n_teacher,'task_config_sha256':sha(config_path or ROOT/'configs'/f'mnist_real_qualified_{kind}.yaml'),'partition_sha256':partition_hash,'no_retrained_checkpoint':True,'not_official_xfedalign':True,
        'communication':ledger,'frozen_config_hash':sha(ROOT/'configs/fagc_frozen_confirmation.json'),
        'n_raw_eval_rows':len(records),'methods':result}
    (output/'audit.json').write_text(json.dumps(summary,indent=2))
    with open(output/'records.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    for m,d in result.items():print(kind,seed,m,'JSD',round(d['jsd'],6),'EDI',round(d['edi'],6),'topk',round(d['topk'],3),flush=True)
    return summary

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--kind',required=True,choices=['none','rotation','patch']);p.add_argument('--config',default=None);p.add_argument('--checkpoint',default=None);p.add_argument('--training-rounds',type=int,default=None);p.add_argument('--teacher',type=int,default=12);p.add_argument('--seed',required=True,type=int);p.add_argument('--replica',type=int,default=0);p.add_argument('--eval',type=int,default=24);p.add_argument('--threads',type=int,default=2);p.add_argument('--out',default='fagc_confirmation_results');a=p.parse_args()
  torch.set_num_threads(a.threads)
  v=verify_idx(ROOT/'data');assert v['status']=='PASS' and v['train']['rows']==60000 and v['test']['rows']==10000
  run(a.kind,a.seed,a.replica,a.eval,ROOT/a.out,a.config,a.checkpoint,a.training_rounds,a.teacher)
