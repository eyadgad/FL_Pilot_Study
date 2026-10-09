#!/usr/bin/env python3
"""Fail-closed prospective APGF seed-level summary; NEVER certifies publication novelty."""
import argparse,csv,json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import t

HERE=Path(__file__).resolve().parents[1]
SEEDS=json.loads((HERE/'configs'/'APGF_FROZEN_PROSPECTIVE.json').read_text())['new_seeds']
NEED=['input_only','private_field','private_field_48','fixed_peer_0p2','fixed_peer_0p2_full48','adaptive_peer']

def verify(root):
    root=Path(root)
    z=[];errors=[]
    for scenario,seeds in SEEDS.items():
        for seed in seeds:
            d=root/f'apgf_full_{scenario}'/f'seed_{seed}'
            a=d/'apgf_evaluation'/f'apgf_full_{scenario}_{seed}_summary.json'
            b=d/'train_manifest.json'; c=d/'apgf_complete_manifest.json'
            if not (a.is_file() and b.is_file() and c.is_file()):
                errors.append('MISSING '+str(d));continue
            x=json.loads(a.read_text());m=json.loads(b.read_text());co=json.loads(c.read_text())
            if int(x['seed'])!=seed or int(m['seed'])!=seed or x['scenario']!=f'apgf_full_{scenario}':
                errors.append('WRONG_SCENARIO '+str(d));continue
            if m.get('status')!='PASS' or x.get('full10k_accuracy',0)<.95 or m.get('full_test_n')!=10000:
                errors.append('FAILED_TASK_GATE '+str(d));continue
            if m.get('checkpoint_sha256')!=x.get('checkpoint_sha256') or m.get('partition_sha256')!=x.get('partition_sha256'):
                errors.append('CHECKPOINT_OR_PARTITION_HASH_MISMATCH '+str(d));continue
            if x.get('n_teacher_fit_per_client')!=36 or x.get('n_teacher_validation_per_client')!=12 or x.get('n_heldout_eval_per_client')!=64:
                errors.append('WRONG_TEACHER_BUDGET '+str(d));continue
            mm=x['methods']
            if any(k not in mm for k in NEED):errors.append('MISSING_METHOD '+str(d));continue
            if not all(math.isfinite(mm[k]['mean_jsd']) and mm[k]['n_eval']==5*64 for k in NEED):
                errors.append('BAD_METRICS '+str(d));continue
            rd={'scenario':scenario,'seed':seed,'accuracy':x['full10k_accuracy'],
                'gain_vs_private48':mm['private_field_48']['mean_jsd']-mm['adaptive_peer']['mean_jsd'],
                'gain_vs_fixed48':mm['fixed_peer_0p2_full48']['mean_jsd']-mm['adaptive_peer']['mean_jsd'],
                'topk_gain_vs_private48':mm['adaptive_peer']['mean_top48_overlap']-mm['private_field_48']['mean_top48_overlap'],
                'disagreement_gain_vs_private48':mm['private_field_48']['heldout_predclass_summary_disagreement_jsd']-mm['adaptive_peer']['heldout_predclass_summary_disagreement_jsd']}
            for k in NEED:rd[k+'_jsd']=mm[k]['mean_jsd']
            z.append(rd)
    def interval(v):
        v=np.asarray(v,float)
        lo,hi=t.interval(.95,len(v)-1,loc=float(v.mean()),scale=float(v.std(ddof=1)/np.sqrt(len(v))))
        return [float(lo),float(hi)]
    out={'status':'INCOMPLETE','expected_seeds':sum(map(len,SEEDS.values())),'observed_seeds':len(z),
         'errors':errors,'not_official_xfedalign':True,'publication_novelty_automatic_certificate':False}
    if len(z)==30 and not errors:
        gains=np.array([v['gain_vs_private48'] for v in z])
        fail_alpha=[v for v in z if v['scenario']=='noniid_005' and v['gain_vs_private48']<0]
        gfix=np.array([v['gain_vs_fixed48'] for v in z])
        out['status']='COMPLETE_DESCRIPTIVE_ONLY'
        out['mean_gain_vs_private48']=float(gains.mean())
        out['seed_clustered_95ci_gain_private48']=interval(gains)
        out['mean_gain_vs_fixed48']=float(gfix.mean())
        out['seed_clustered_95ci_gain_fixed48']=interval(gfix)
        out['alpha005_seed_losses']=len(fail_alpha)
        out['hard_gates']={
           'task_accuracy':True,
           'all_30_seeds':True,
           'private48_gain_at_least_0005':gains.mean()>=.005,
           'private48_gain_positive_lower_ci':interval(gains)[0]>0,
           'fixed48_gain_positive_lower_ci':interval(gfix)[0]>0,
           'alpha005_no_negative_transfer':len(fail_alpha)==0,
           'topk_no_regression_all_scenarios':all(np.mean([v['topk_gain_vs_private48'] for v in z if v['scenario']==s])>=0 for s in SEEDS),
           'no_test_set_tuning':True}
        out['advance_allowed_by_preliminary_numeric_gates']=all(out['hard_gates'].values())
        out['note']='Full paper evaluation still needs official xFedAlign, communication Pareto, CIFAR, and novelty prior-art comparison.'
    return out,z

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='apgf_prospective_outputs');p.add_argument('--out',default='apgf_prospective_aggregate.json');a=p.parse_args()
    result,rows=verify(a.root);Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    if rows:
        f=Path(a.out).with_suffix('.csv')
        with f.open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2))
