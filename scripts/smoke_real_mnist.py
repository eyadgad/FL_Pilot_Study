#!/usr/bin/env python3
"""Real-MNIST-only APGF engineering smoke. No model training, no data downloads.
Uses supplied v5 model checkpoint and disjoint real-MNIST client partitions.
Validation only selects lambda; evaluation is never examined until frozen.
"""
from __future__ import annotations
import argparse,sys,hashlib,json,csv
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def records(model,ds,count,ig_steps,thread_seed):
    from ucpa_fl.explain import integrated_gradients
    it=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0)
    out=[]
    for x,_ in it:
        if len(out)>=count:break
        with torch.no_grad(): p=model(x).argmax(1)
        o=integrated_gradients(model,x,p,steps=ig_steps).flatten(1)
        o=(o.clamp_min(0)/(o.sum(1,keepdim=True)+1e-12)).detach().numpy()
        for k in range(x.shape[0]):
            if len(out)>=count:break
            out.append({'x':x[k].numpy().copy(),'pred':int(p[k]),'oracle_map':o[k].copy()})
    if len(out)!=count:raise ValueError('Insufficient real-MNIST records')
    return out

def run(root,experiment,seed,settings,outdir,config_override=None,checkpoint_override=None,manifest_override=None):
    root=Path(root).resolve();outdir=Path(outdir).resolve()
    sys.path.insert(0,str(root));sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from ucpa_fl.config import load_config
    from ucpa_fl.datasets import load_data,make_loader
    from ucpa_fl.models import MNISTCNN
    from ucpa_fl.federated import evaluate
    from ucpa_fl.fagc import fagc_map
    from ucpa_fl.metrics import topk_overlap, normalize_np
    from ucpa_fl.alignment import jsd
    from src.apgf import fit_gated_fields,quantized_teacher_stats,candidate_fields
    from scripts.verify_real_mnist import verify
    torch.set_num_threads(3)
    assert verify(root/'data')['status']=='PASS'
    kind=experiment
    cfgpath=Path(config_override).resolve() if config_override else root/'configs'/f'mnist_real_qualified_{kind}.yaml'
    cfg=load_config(cfgpath);assert not cfg.dataset.download and cfg.dataset.name=='mnist'
    if config_override and seed not in cfg.seeds:raise ValueError('Seed not preregistered in the chosen frozen YAML')
    if cfg.shift.kind!=kind:raise ValueError('Requested shift differs from config')
    cfg.dataset.root=str(root/'data');cfg.dataset.test_limit=0
    bundle=load_data(cfg,int(seed))
    checkpoint=Path(checkpoint_override).resolve() if checkpoint_override else root/'fagc_independent_checkpoints'/kind/str(seed)/'global_task_model.pt'
    manifest=json.loads(Path(manifest_override).read_text() if manifest_override else (checkpoint.parent/'manifest.json').read_text())
    assert sha(checkpoint)==manifest['checkpoint_sha256']
    assert manifest['seed']==seed and (manifest.get('kind',kind)==kind) and manifest.get('passes_gate',manifest.get('status')=='PASS')
    if len(bundle.test)!=10000:raise ValueError('No full real MNIST test split')
    from scripts.run_fagc_full_suite import _partition_hash
    assert _partition_hash(bundle)==manifest['partition_sha256'], 'Disjoint partition changed'
    model=MNISTCNN();model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'));model.eval()
    acc=evaluate(model,make_loader(bundle.test,256,False,seed,0),torch.device('cpu'))['accuracy']
    if acc<settings['checkpoint_accuracy_gate']:raise ValueError('Below accuracy floor, no IG permitted')
    print(kind,seed,'qualified',round(acc,5),flush=True)
    bundle.set_round(manifest['rounds']-1)
    nfit=settings['teacher_fit_per_client'];nval=settings['teacher_validation_per_client'];neval=settings['final_heldout_eval_per_client']
    teachers=[];evals=[]
    for i in range(cfg.federation.n_clients):
        t=records(model,bundle.calibration_clients[i],nfit+nval,settings['ig_steps'],seed)
        # CALIBRATION and VALIDATION separate; no teacher/eval overlap via data partition
        teachers.append((t[:nfit],t[nfit:]))
        evals.append(records(model,bundle.eval_clients[i],neval,settings['ig_steps'],seed))
        print(kind,seed,'client',i,'fit/val/eval',nfit,nval,neval,flush=True)
    fitted=fit_gated_fields([t[0] for t in teachers],[t[1] for t in teachers],tuple(settings['peer_candidates']),
                            improvement_margin=settings['validation_min_improvement_jsd'],
                            min_pair_win_fraction=settings['validation_min_fraction_positive'],
                            power=settings['teacher_input_power'])
    # Strong competing controls use all 48 teacher records.  The adaptive method
    # is deliberately evaluated at 36 field-fit + 12 selection; it is NOT allowed
    # to claim superiority over a 36-only private control if private-48 wins.
    all48,_=candidate_fields([quantized_teacher_stats(t[0]+t[1]) for t in teachers],tuple(settings['peer_candidates']))
    strong_private=np.stack([f[0.] for f in all48]);strong_fixed=np.stack([f[.2] for f in all48])
    method_fields={'private_field':fitted['private'],'private_field_48':strong_private,
                   'fixed_peer_0p2':fitted['fixed_peer_0p2'], 'fixed_peer_0p2_full48':strong_fixed,
                   'adaptive_peer':fitted['selected']}
    methods=list(settings['evaluated_methods'])
    rows=[];summary={}
    for i,client in enumerate(evals):
        for j,r in enumerate(client):
            ref=normalize_np(np.asarray(r['oracle_map'])[None])[0]
            mapped={'input_only':normalize_np(np.maximum(r['x'].reshape(-1),0)[None]**settings['teacher_input_power'])[0]}
            for name,field in method_fields.items():
                mapped[name]=fagc_map(r['x'],np.ones(784),field[i],power=settings['teacher_input_power'],class_exponent=0)
            row={'seed':seed,'kind':kind,'client':i,'local_index':j,'pred_class':r['pred']}
            for m in methods:
                row[m+'_jsd']=float(jsd(mapped[m],ref));row[m+'_topk']=float(topk_overlap(mapped[m],ref,48))
            rows.append(row)
    for m in methods:
        v=np.asarray([r[m+'_jsd'] for r in rows]);t=np.asarray([r[m+'_topk'] for r in rows]);
        summary[m]={'mean_jsd':float(v.mean()),'mean_top48_overlap':float(t.mean()),'n_eval':len(v)}
    # conservative no-force definition: EDI-like comparison on disjoint evaluation maps,
    # NOT directly comparable with published artifacts-based EDI.
    from ucpa_fl.metrics import pairwise_edi
    for m in methods:
        by=np.zeros((len(evals),10,784));counts=np.zeros((len(evals),10),int)
        for i,recs in enumerate(evals):
            for c in range(10):
                rr=[r for r in recs if r['pred']==c]
                if not rr:continue
                counts[i,c]=len(rr)
                mats=[]
                for r in rr:
                    if m=='input_only':p=normalize_np(np.maximum(r['x'].reshape(-1),0)[None]**settings['teacher_input_power'])[0]
                    else:p=fagc_map(r['x'],np.ones(784),method_fields[m][i],power=settings['teacher_input_power'],class_exponent=0)
                    mats.append(p)
                by[i,c]=normalize_np(np.mean(mats,axis=0)[None])[0]
        summary[m]['heldout_predclass_summary_disagreement_jsd']=float(pairwise_edi(by,counts))
    outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/f'{cfg.experiment_name}_{seed}_records.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'kind':kind,'scenario':cfg.experiment_name,'seed':seed,'status':'EXPLORATORY','dataset_provenance':'genuine CSV-origin MNIST via v5 verified IDX',
            'checkpoint_sha256':sha(checkpoint),'partition_sha256':manifest['partition_sha256'],'config_sha256':sha(cfgpath),
            'protocol_sha256':sha(Path(__file__).resolve().parents[1]/'configs'/'smoke_predeclared.json'),
            'full10k_accuracy':acc,'n_teacher_fit_per_client':nfit,'n_teacher_validation_per_client':nval,
            'n_heldout_eval_per_client':neval,'n_clients':len(evals), 'all_data_partitions_disjoint':True,
            'methods':summary,'selected_gates':fitted['decisions'],'ledger':fitted['ledger'],
            'notes':['Already-trained CNNs were reused; this is not new independent FedAvg training.',
                     'The decision rule was designed after looking at previous studies; this is exploratory, not publishable confirmation.',
                     'Strong private-only and fixed-sharing controls also fit with all 48 available teachers.',
                     'No class-conditioning or xFedAlign surrogate included in this focused peer-gain smoke.',
                     'Validation records never enter fit statistics or final evaluation.']}
    (outdir/f'{cfg.experiment_name}_{seed}_summary.json').write_text(json.dumps(report,indent=2))
    print('SUMMARY',kind,seed,json.dumps({k:round(v['mean_jsd'],6) for k,v in summary.items()}),'decisions',[x['selected_weight'] for x in fitted['decisions']],flush=True)
    return report

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--v5-root',required=True);pa.add_argument('--seed',type=int,required=True);pa.add_argument('--kind',choices=['patch','rotation'],required=True);pa.add_argument('--out',default='outputs')
    a=pa.parse_args();cfg=json.loads((Path(__file__).resolve().parents[1]/'configs'/'smoke_predeclared.json').read_text())
    if {'kind':a.kind,'seed':a.seed} not in cfg['eval_seeds']:raise SystemExit('Seed not in predeclared smoke matrix')
    run(a.v5_root,a.kind,a.seed,cfg,a.out)
