#!/usr/bin/env python3
"""Full 5-client FedAvg -> freeze CNN -> real IG -> paired QF-SCAD study.
All scientific configs have immutable SHA256 entries. This runs exactly one seed.
No image downloads, no fallback synthetic data, no reuse of test maps for fitting.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,os,argparse,hashlib,json,csv,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from ucpa_fl.config import load_config
from ucpa_fl.datasets import load_data,make_loader
from ucpa_fl.federated import train_fedavg,train_fedprox,evaluate
from ucpa_fl.repro import set_seed,resolve_device,environment_snapshot
from study_v7.data import verify_data,collect,save_splits,sha
from study_v7.codec import norm,fit_fields
from study_v7.correction import train,train_private,predict
from study_v7.baselines import static_map,fedattr_class_prior
from study_v7.apgf_gate import choose_apgf
from study_v7.metrics import jsd,topk,disagreement,functional_auc
from study_v7.published_baselines import collect_published_controls

class Logger:
    def __init__(self,run_dir):
        self.run_dir=Path(run_dir);(self.run_dir/'checkpoints').mkdir(parents=True,exist_ok=False)
        self.stream=(self.run_dir/'rounds.jsonl').open('w')
    def metric(self,name,value,**tags):
        self.stream.write(json.dumps({'type':'metric','name':name,'value':float(value),**tags})+'\n');self.stream.flush()
    def event(self,name,**tags):
        self.stream.write(json.dumps({'type':'event','name':name,**tags})+'\n');self.stream.flush()
    def close(self):self.stream.close()


def index_checked_hw(x,dataset):
    return (28,28) if dataset=='mnist' else (32,32)


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n')


def run_one(config,seed,root,outroot,device='auto',threads=4):
    start=time.perf_counter()
    config=Path(config).resolve();root=Path(root).resolve();outroot=Path(outroot).resolve()
    expected=json.loads((ROOT/'configs'/'EXPANDED_CONFIG_HASHES.json').read_text())
    if config.parent!=(ROOT/'configs'/'expanded').resolve() or config.name not in expected:
        raise ValueError('Config outside frozen prospective matrix')
    if sha(config)!=expected[config.name]:raise ValueError('Frozen config changed')
    spec=json.loads((ROOT/'configs'/'EVALUATION_PROTOCOL.json').read_text())
    matrix=json.loads((ROOT/'configs'/'EXPANDED_STUDY_PLAN.json').read_text())['matrix'][config.name]
    cfg=load_config(config)
    if seed not in cfg.seeds or cfg.dataset.name not in ('mnist','cifar10'):
        raise ValueError('Invalid prospective dataset/seed')
    if cfg.dataset.download or cfg.dataset.train_limit or cfg.dataset.test_limit:
        raise ValueError('Non-offline or truncated source forbidden')
    if (cfg.federation.n_clients,cfg.federation.rounds,cfg.federation.local_epochs)!=(matrix['n_clients'],matrix['rounds'],matrix['local_epochs']):
        raise ValueError('Frozen client/round/epoch factors changed')
    if cfg.tags.get('task_optimizer','fedavg')!=matrix['optimizer']:
        raise ValueError('Frozen optimizer differs')
    if tuple(spec['datasets'][cfg.dataset.name]['shape'])!=((1,28,28) if cfg.dataset.name=='mnist' else (3,32,32)):
        raise ValueError('Incorrect dataset specification')
    torch.set_num_threads(int(threads));dev=resolve_device(device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise ValueError('Requested CUDA unavailable')
    provenance=verify_data(cfg.dataset.name,root)
    if provenance['train_count']!=spec['datasets'][cfg.dataset.name]['train'] or provenance['test_count']!=10000:
        raise ValueError('Original dataset counts failed')
    cfg.dataset.root=str(root)
    set_seed(int(seed),True)
    bundle=load_data(cfg,int(seed))
    if len(bundle.test)!=10000 or bundle.input_shape!=tuple(spec['datasets'][cfg.dataset.name]['shape']):
        raise ValueError('Dataset shape/test-count mismatch')
    hw=tuple(bundle.input_shape[1:])
    nfit=int(spec['teachers_per_client']);neval=int(spec['evaluation_per_client'])
    sizes=[{'client':i,'task':len(bundle.task_clients[i]),'surrogate':len(bundle.surrogate_clients[i]),
            'artifact':len(bundle.artifact_clients[i]),'teacher':len(bundle.calibration_clients[i]),
            'evaluation':len(bundle.eval_clients[i])} for i in range(cfg.federation.n_clients)]
    if any(s['teacher']<nfit or s['evaluation']<neval for s in sizes):
        raise ValueError(f'Insufficient genuine real-data records; do not pad or synthesize: {sizes}')
    p=outroot/cfg.experiment_name/f'seed_{seed}'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.mkdir(exist_ok=False) # never overwrite an earlier experiment
    save_json(p/'dataset_provenance.json',provenance)
    splitsha,split_counts=save_splits(bundle,p/'partition_indices.npz')
    save_json(p/'environment.json',environment_snapshot())
    (p/'config.yaml').write_bytes(config.read_bytes())
    runmanifest={'status':'TRAINING','dataset':cfg.dataset.name,'scenario':cfg.experiment_name,'seed':int(seed),
       'config_sha256':sha(config),'dataset_file_sha256':provenance['file_sha256'],
       'partition_file_sha256':splitsha,'partition_counts':split_counts,
       'model':'MNISTCNN' if cfg.dataset.name=='mnist' else 'CIFARSmallCNN',
       'n_original_train':provenance['train_count'],'n_original_test':10000,
       'training_rounds':cfg.federation.rounds,'local_epochs':cfg.federation.local_epochs,
       'device':str(dev),'task_optimizer':matrix['optimizer'],'candidate_method':'QF-SCAD v7 (77 parameters, quantized FedAvg)'}
    save_json(p/'train_manifest.json',runmanifest)
    logger=Logger(p)
    try:
        model,task=(train_fedprox(cfg,bundle,int(seed),dev,logger,mu=0.01) if matrix['optimizer']=='fedprox' else train_fedavg(cfg,bundle,int(seed),dev,logger))
    finally:logger.close()
    runmanifest.update(task_accuracy_full10k=float(task['accuracy']),task_test_n=int(task['n']),
        checkpoint_sha256=sha(p/'checkpoints'/'global_task_model.pt'),cnn_training_seconds=time.perf_counter()-start)
    floor=spec['datasets'][cfg.dataset.name]['accuracy_gate']
    if task['accuracy']<floor or task['n']!=10000:
        runmanifest['status']='FAIL_TASK_GATE';save_json(p/'train_manifest.json',runmanifest)
        return runmanifest
    runmanifest['status']='PASS';save_json(p/'train_manifest.json',runmanifest)
    # The task CNN weights and checkpoint must remain identical for every method.
    model.eval();bundle.set_round(cfg.federation.rounds-1)
    teachers=[];held=[]
    for i in range(cfg.federation.n_clients):
        teachers.append(collect(model,bundle.calibration_clients[i],nfit,dev,spec['ig_midpoint_steps'],hw,
                                batch_size=spec['ig_batch_size']))
        held.append(collect(model,bundle.eval_clients[i],neval,dev,spec['ig_midpoint_steps'],hw,
                            batch_size=spec['ig_batch_size']))
    X=[t['x'] for t in teachers];Y=[t['ig'] for t in teachers]
    fields,staticledger=fit_fields(X,Y,hw,peer_weight=spec['peer_weight'],power=spec['input_power'])
    # Match teacher budgets and architecture; private corrector receives all 48 images.
    corr_seed=int(seed)+int(spec['explanation_seed_offset'])
    common=dict(rounds=int(spec['qf_rounds']),steps_per_round=int(spec['qf_local_steps']),lr=float(spec['qf_lr']))
    net,neuralledger,round_losses=train(X,Y,fields['fixed'],hw,corr_seed,device=dev,**common)
    private_nets=train_private(X,Y,fields['private'],hw,corr_seed,
                               steps=common['rounds']*common['steps_per_round'],lr=common['lr'],device=dev)
    peerprivate_nets=train_private(X,Y,fields['fixed'],hw,corr_seed,
                               steps=common['rounds']*common['steps_per_round'],lr=common['lr'],device=dev)
    # Architecture ablation: federated convolution alone with a neutral gain.
    neutral=np.ones_like(fields['fixed'])
    net_noprior,ablation_ledger,_=train(X,Y,neutral,hw,corr_seed,device=dev,**common)
    priors,classledger=fedattr_class_prior(teachers)
    apgf_fields,apgf_decisions,apgf_ledger=choose_apgf(teachers,hw)
    methods={}
    for i,d in enumerate(held):
        x=d['x']
        controls=collect_published_controls(model,bundle.eval_clients[i],neval,dev)
        methods_i={
          'gradient_saliency':controls['gradient_saliency'],
          'gradcam':controls['gradcam'],
          'input_only':norm(np.maximum(x,0)**spec['input_power']),
          'private_field_48':static_map(x,fields['private'][i],spec['input_power']),
          'fixed_fed_field_48':static_map(x,fields['fixed'][i],spec['input_power']),
          'global_field_48':static_map(x,fields['global'][i],spec['input_power']),
          'apgf_v6_gate_36fit_12val':static_map(x,apgf_fields[i],spec['input_power']),
          'fedattr_class_template_proxy':np.stack([priors[int(c)] for c in d['pred']]),
          'private_conv_48':predict(private_nets[i],x,fields['private'][i],hw),
          'private_conv_with_fed_prior_48':predict(peerprivate_nets[i],x,fields['fixed'][i],hw),
          'federated_conv_without_prior':predict(net_noprior,x,neutral[i],hw),
          'qfscad_v7':predict(net,x,fields['fixed'][i],hw),
        }
        for m,v in methods_i.items():methods.setdefault(m,[]).append(v)
    if not all(len(a)==cfg.federation.n_clients for a in methods.values()):raise ValueError('Method/client mismatch')
    # Never reduce to pooled pixels for experimental significance: save each true seed/client/example.
    rows=[]
    for i,d in enumerate(held):
        for j in range(neval):
            row={'dataset':cfg.dataset.name,'scenario':cfg.experiment_name,'seed':int(seed),'client':i,
                 'record':j,'predicted_class':int(d['pred'][j])}
            for m,arrays in methods.items():
                row[m+'_jsd']=jsd(arrays[i][j],d['ig'][j]);row[m+'_topk']=topk(arrays[i][j],d['ig'][j],spec['topk_spatial'][cfg.dataset.name])
            rows.append(row)
    with (p/'heldout_per_example.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    evalstats={}
    for m,v in methods.items():
        meanjsd=float(np.mean([r[m+'_jsd'] for r in rows]));meantop=float(np.mean([r[m+'_topk'] for r in rows]))
        evalstats[m]={'mean_oracle_jsd':meanjsd,'mean_topk':meantop,
            'heldout_predclass_summary_disagreement':disagreement(v,[z['pred'] for z in held])}
    # All methods share identical true images and task-model class targets for perturbation.
    fnrows=[];nfunc=int(spec['functional_records_per_client'])
    for i in range(cfg.federation.n_clients):
        raw_loader=DataLoader(bundle.eval_clients[i],batch_size=1,shuffle=False,num_workers=0)
        for j,(real_x,_) in enumerate(raw_loader):
            if j>=nfunc:break
            target=int(held[i]['pred'][j])
            row={'dataset':cfg.dataset.name,'scenario':cfg.experiment_name,'seed':int(seed),'client':i,'record':j}
            for m,v in methods.items():
                deletion,insertion=functional_auc(model,real_x[0],target,v[i][j],dev,steps=spec['functional_steps'])
                row[m+'_deletion_auc']=deletion;row[m+'_insertion_auc']=insertion
            fnrows.append(row)
    with (p/'functional_per_example.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(fnrows[0]));writer.writeheader();writer.writerows(fnrows)
    for m in methods:
        evalstats[m].update(mean_deletion_auc=float(np.mean([r[m+'_deletion_auc'] for r in fnrows])),
                            mean_insertion_auc=float(np.mean([r[m+'_insertion_auc'] for r in fnrows])))
    # Serialize candidate's initial field and correction rounds. Count BOTH upload and download.
    wire_q=[s+n for s,n in zip(staticledger['static_total_bytes_per_client'],neuralledger['model_total_bytes'])]
    wire_noprior=ablation_ledger['model_total_bytes']
    ledgers={'static_fields':staticledger,'qfscad_correction':neuralledger,
             'qfscad_bytes_per_client':wire_q,'qfscad_mean_bytes_per_client':float(np.mean(wire_q)),
             'federated_conv_without_prior_bytes_per_client':wire_noprior,
             'private_conv_bytes_per_client':[0]*cfg.federation.n_clients,
             'private_conv_with_fed_prior_bytes_per_client':staticledger['static_total_bytes_per_client'],
             'fedattr_template_proxy':classledger,
             'apgf_v6_generalized':apgf_ledger,
             'reported_bytes_are_exact_packet_payloads_not_transport_overhead':True}
    save_json(p/'communication_ledger.json',ledgers)
    save_json(p/'apgf_gate_decisions.json',apgf_decisions)
    save_json(p/'explanation_round_losses.json',round_losses)
    save_json(p/'explanation_metrics.json',evalstats)
    info={'status':'COMPLETE','seed':int(seed),'dataset':cfg.dataset.name,'scenario':cfg.experiment_name,
          'accuracy_full10k':task['accuracy'],'checkpoint_sha256':runmanifest['checkpoint_sha256'],
          'partition_file_sha256':splitsha,'config_sha256':sha(config),
          'teacher_fit_per_client':nfit,'heldout_eval_per_client':neval,
          'direct_ig_steps':spec['ig_midpoint_steps'],'functional_records_per_client':nfunc,
          'methods':evalstats,'communication':ledgers,'runtime_seconds':time.perf_counter()-start,
          'scientific_status':'PROSPECTIVE_RAW_RESULT_NOT_INDEPENDENTLY_CONFIRMED',
          'baseline_limitations':'Published: saliency and Grad-CAM independent reproductions. FedProx separate task optimizer. FedAttr class template is a PROXY, not official xFedAlign/FedAttr-Agg; SOTA comparison remains INCOMPLETE.'}
    save_json(p/'complete_manifest.json',info)
    return info

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--data-root',default='data')
    parser.add_argument('--output-root',default='full_study_outputs')
    parser.add_argument('--device',default='auto',help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--threads',type=int,default=4)
    a=parser.parse_args()
    out=run_one(a.config,a.seed,a.data_root,a.output_root,a.device,a.threads)
    print(json.dumps({'status':out['status'],'dataset':out['dataset'],'scenario':out['scenario'],
                       'seed':out['seed'],'accuracy':out.get('accuracy_full10k',out.get('task_accuracy_full10k'))},indent=2))
