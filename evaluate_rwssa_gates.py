#!/usr/bin/env python3
"""Frozen pilot and full confirmation gates; no incomplete-evidence passes.

This is a decision aid, not an estimator of conference acceptance probability.
"""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import yaml
import numpy as np
ROOT=Path(__file__).resolve().parent
LO=['sample_fidelity_jsd','pairwise_edi','deletion_auc']
HI=['insertion_auc','topk_oracle_overlap']
METRICS=LO+HI
BASELINES=['xfedalign_median','fedattr_mean','iflash_proxy','xfedalign_beta_0p4']
ABLATIONS=['rwssa_binary','rwssa_no_safety','rwssa_no_calibration','rwssa_class_scalar']

def run_gate(tier,filename,out):
    man=yaml.safe_load((ROOT/'configs/suite_manifest.yaml').read_text())
    items=[i for i in man['configs'] if i['tier']==tier]
    with open(filename,encoding='utf8',newline='') as f: rows=list(csv.DictReader(f))
    by={(r['experiment'],int(r['seed']),r['scenario'],r['method']):r for r in rows}
    missing=[];evals=[]
    for item in items:
        cfg=yaml.safe_load((ROOT/'configs'/item['config']).read_text())
        for seed in cfg['seeds']:
            for method in cfg['alignment']['methods']:
                if (item['experiment_name'],seed,'clean',method) not in by:
                    missing.append([item['experiment_name'],seed,method])
    if missing or not items:
        result={'tier':tier,'decision':'PENDING','expected_configurations':len(items),'missing_count':len(missing),'missing_sample':missing[:40]}
        Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result
    for item in items:
        cfg=yaml.safe_load((ROOT/'configs'/item['config']).read_text())
        exp=item['experiment_name'];byseed=[];task=[];comp={m:[] for m in METRICS}
        for seed in cfg['seeds']:
            a=by[(exp,seed,'clean','rwssa')]
            task.append(float(a.get('client_task_accuracy_mean','nan')))
            win={}
            for metric in METRICS:
                val=float(a[metric]);others=[float(by[(exp,seed,'clean',b)][metric]) for b in BASELINES]
                if not np.isfinite(val) or not np.isfinite(others).all(): raise ValueError(f'non-finite {exp} {seed} {metric}')
                # meaningful small tolerance of 1e-6 only to avoid rounding ties
                win[metric]=bool(val<min(others)-1e-6) if metric in LO else bool(val>max(others)+1e-6)
                comp[metric].append(float(val-(min(others) if metric in LO else max(others))))
            byseed.append(win)
        n=len(byseed)
        mean_scores={k:float(np.mean(v)) for k,v in comp.items()}
        mean_wins={k:int(sum(r[k] for r in byseed)) for k in METRICS}
        kstrong=sum(mean_wins[m]>= (2 if tier=='P' else 7) for m in METRICS)
        x=[]
        for seed in cfg['seeds']:
            a=by[(exp,seed,'clean','rwssa')];loc=by[(exp,seed,'clean','local')]
            x.append({'seed':seed,
               'jsd_delta_local':float(a['sample_fidelity_jsd'])-float(loc['sample_fidelity_jsd']),
               'deletion_delta_local':float(a['deletion_auc'])-float(loc['deletion_auc']),
               'insertion_delta_local':float(a['insertion_auc'])-float(loc['insertion_auc']),
               'topk_delta_local':float(a['topk_oracle_overlap'])-float(loc['topk_oracle_overlap']),
               'extra_comm_factor':float(a['communication_bytes_per_client_artifact'])/max(float(by[(exp,seed,'clean','xfedalign_median')]['communication_bytes_per_client_artifact']),1)})
        gap={k:float(np.mean([r[k] for r in x])) for k in x[0] if k!='seed'}
        valid=bool(min(task)>= (0.35 if tier=='P' else 0.50))
        sound=valid and gap['jsd_delta_local']<=0.03 and gap['deletion_delta_local']<=0.03 and gap['insertion_delta_local']>=-0.03 and gap['topk_delta_local']>=-0.06
        # Requirement applies to each shift family; harsh enough to kill rather than oversell.
        wins_vs_xfed=sum(1 for m in METRICS if sum(
            (float(by[(exp,s,'clean','rwssa')][m])<float(by[(exp,s,'clean','xfedalign_median')][m])-1e-6) if m in LO else
            (float(by[(exp,s,'clean','rwssa')][m])>float(by[(exp,s,'clean','xfedalign_median')][m])+1e-6)
            for s in cfg['seeds'])>= (2 if tier=='P' else 7))
        evals.append({'experiment':exp,'n_seeds':n,'task_acc_min':min(task),'soundness':sound,'strongest_baseline_metric_wins':mean_wins,
                      'mean_gap_to_strongest':mean_scores,'wins_vs_xfed_metrics':wins_vs_xfed,'communication_and_local_deltas':gap,
                      'qualifies_family':bool(sound and wins_vs_xfed>=4 and gap['extra_comm_factor']<=12.0)})
    # Gate criteria frozen before main runs. Pilot: 2/3; full: 5/6 heterogeneous + 1 IID control noncollapse.
    relevant=[x for x in evals if not ('_none' in x['experiment'])]
    good=sum(x['qualifies_family'] for x in relevant)
    needed=2 if tier=='P' else 4
    # Strict ablation gate: averaged RWSSA JSD must not be worse than both simple ablations everywhere.
    ablation_wins=0;ablation_comparisons=0
    for item in items:
        exp=item['experiment_name'];cfg=yaml.safe_load((ROOT/'configs'/item['config']).read_text())
        for seed in cfg['seeds']:
            a=float(by[(exp,seed,'clean','rwssa')]['sample_fidelity_jsd'])
            for method in ('rwssa_binary','rwssa_no_safety'):
                b=float(by[(exp,seed,'clean',method)]['sample_fidelity_jsd'])
                ablation_wins+=int(a<b-1e-6);ablation_comparisons+=1
    # This measure is descriptive for pilot; full test requires >50% superiority.
    ablations_ok=bool(ablation_wins>=int(np.ceil(.5*ablation_comparisons)))
    approved=good>=needed and (tier=='P' or ablations_ok)
    result={'tier':tier,'decision':('AUTHORIZE_FULL_STUDY' if approved else 'STOP_AND_REVIEW') if tier=='P' else ('CONTINUE_RESEARCH' if approved else 'REVISE_OR_KILL'),
            'gate':{'qualifying_families':good,'required':needed,'ablation_wins':ablation_wins,'ablation_comparisons':ablation_comparisons,'ablations_pass':ablations_ok},
            'families':evals,'caveats':['Approximated xFedAlign/iFLASH controls are not official author code.',
             'Selecting λ via two-metric JSD/top-k risk does not guarantee deletion/insertion risk.',
             'Communication includes unprotected dense client gain vectors.']}
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(result,indent=2));print(json.dumps({'decision':result['decision'],'gate':result['gate']},indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tier',choices=['P','A'],required=True);p.add_argument('--runs',default='aggregate_all/runs.csv');p.add_argument('--out',default='aggregate_all/rwssa_gate.json');a=p.parse_args()
    run_gate(a.tier,a.runs,a.out)
