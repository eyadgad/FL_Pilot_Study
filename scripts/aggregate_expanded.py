#!/usr/bin/env python3
"""Recompute all saved metrics; no unverifiable promotion and no pseudo-replication."""
from __future__ import annotations
import json,csv,argparse,hashlib
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]
METHODS=['input_only','private_field_48','fixed_fed_field_48','global_field_48','apgf_v6_gate_36fit_12val',
   'fedattr_class_template_proxy','private_conv_48','private_conv_with_fed_prior_48',
   'federated_conv_without_prior','qfscad_v7','gradient_saliency','gradcam']

def evaluate(root):
    plan=json.loads((ROOT/'configs/EXPANDED_STUDY_PLAN.json').read_text())['matrix']
    source=[];gaps=[];bad=[]
    for config,setting in plan.items():
        name=Path(config).stem
        for seed in setting['seeds']:
            p=root/name/f'seed_{seed}'
            complete=p/'complete_manifest.json';train=p/'train_manifest.json'
            if not complete.exists():
                if train.exists():
                    tr=json.loads(train.read_text())
                    if tr.get('status')=='FAIL_TASK_GATE':bad.append({'config':name,'seed':seed,'status':'FAIL_TASK_GATE'});continue
                gaps.append({'config':name,'seed':seed});continue
            d=json.loads(complete.read_text())
            if d.get('status')!='COMPLETE' or d.get('seed')!=seed or d.get('scenario')!=name:raise ValueError(f'Bad run manifest {complete}')
            raw=list(csv.DictReader((p/'heldout_per_example.csv').open(newline='')))
            fun=list(csv.DictReader((p/'functional_per_example.csv').open(newline='')))
            if len(raw)!=setting['n_clients']*64 or len(fun)!=setting['n_clients']*8:
                raise ValueError(f'Incomplete independent real images in {p}')
            for m in METHODS:
                expected=d['methods'][m]
                exact={
                    'jsd':float(np.mean([float(r[f'{m}_jsd']) for r in raw])),
                    'topk':float(np.mean([float(r[f'{m}_topk']) for r in raw])),
                    'deletion':float(np.mean([float(r[f'{m}_deletion_auc']) for r in fun])),
                    'insertion':float(np.mean([float(r[f'{m}_insertion_auc']) for r in fun]))}
                for rawname,summaryname in [('jsd','mean_oracle_jsd'),('topk','mean_topk'),('deletion','mean_deletion_auc'),('insertion','mean_insertion_auc')]:
                    if not np.isclose(exact[rawname],expected[summaryname],atol=1e-9,rtol=0):
                        raise ValueError(f'Metric recomputation mismatch: {p} {m} {rawname}')
                source.append({'config':name,'dataset':setting['dataset'],'split':setting['split'],
                  'optimizer':setting['optimizer'],'stage':setting['kind'],'n_clients':setting['n_clients'],
                  'rounds':setting['rounds'],'local_epochs':setting['local_epochs'],
                  'seed':seed,'method':m,**exact,'edi_proxy':expected['heldout_predclass_summary_disagreement'],
                  'task_accuracy':d['accuracy_full10k']})
    comparisons=[]
    for conf in sorted(set(r['config'] for r in source)):
        group=[r for r in source if r['config']==conf]
        for base in ('private_conv_48','private_conv_with_fed_prior_48','fixed_fed_field_48',
                     'apgf_v6_gate_36fit_12val','gradient_saliency','gradcam'):
            by={r['seed']:r for r in group if r['method']=='qfscad_v7'}
            bc={r['seed']:r for r in group if r['method']==base}
            ids=sorted(set(by)&set(bc));d=np.array([bc[s]['jsd']-by[s]['jsd'] for s in ids])
            if not len(d):continue
            ci=stats.t.interval(.95,len(d)-1,loc=float(d.mean()),scale=stats.sem(d)) if len(d)>1 and np.std(d,ddof=1)>0 else (float('nan'),float('nan'))
            comparisons.append({'config':conf,'baseline':base,'seeds':len(ids),'jsd_gain':float(d.mean()),
                        'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'positive_seeds':int(sum(d>0))})
    promotion='NOT_ELIGIBLE_OFFICIAL_XFEDALIGN_NOT_PAIRED' # Never auto-promote on proxies.
    return source,comparisons,{'planned_runs':sum(len(z['seeds']) for z in plan.values()),'complete':len(source)//len(METHODS),
      'missing':gaps,'accuracy_gate_failures':bad,'promotion':promotion,
      'warning':'No official xFedAlign paired implementation; custom proxies must not be treated as SOTA reproductions.'}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',default='expanded_results');a.add_argument('--out',default='expanded_analysis')
    x=a.parse_args();out=Path(x.out);out.mkdir(parents=True,exist_ok=True)
    rows,pairs,summary=evaluate(Path(x.root))
    for file,data in [('per_seed_method.csv',rows),('paired_seed_comparisons.csv',pairs)]:
        if data:
            with (out/file).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (out/'overview.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['planned_runs','complete','promotion']},indent=2))
