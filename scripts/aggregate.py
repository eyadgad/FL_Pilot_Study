#!/usr/bin/env python3
"""Immutable-summary reanalysis. Significance uses seed means, never sample pseudo-replication."""
import json,csv,argparse,sys
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]
METHODS=['input_only','apgf_v6_gate_36fit_12val','private_field_48','fixed_fed_field_48','global_field_48',
         'fedattr_class_template_proxy','private_conv_48','private_conv_with_fed_prior_48',
         'federated_conv_without_prior','qfscad_v7']


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',default='full_study_outputs');p.add_argument('--out',default='study_analysis');a=p.parse_args()
    root=Path(a.output_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs'/'PROTOCOL.json').read_text())
    rows=[];missing=[];fail=[]
    for name,seeds in cfg['seed_matrix'].items():
        for seed in seeds:
            path=root/name/f'seed_{seed}'
            complete=path/'complete_manifest.json';tr=path/'train_manifest.json'
            if complete.exists():
                j=json.loads(complete.read_text())
                if j['status']!='COMPLETE' or j['seed']!=seed or j['scenario']!=name:
                    raise ValueError(f'Invalid manifest {complete}')
                # Independent table recalculation from saved per-example source rows.
                with (path/'heldout_per_example.csv').open(newline='') as f:
                    metric_rows=list(csv.DictReader(f))
                with (path/'functional_per_example.csv').open(newline='') as f:
                    functional_rows=list(csv.DictReader(f))
                if len(metric_rows)!=5*cfg['evaluation_per_client'] or len(functional_rows)!=5*cfg['functional_records_per_client']:
                    raise ValueError(f'Missing original held-out evaluation records at {path}')
                for m in METHODS:
                    z=j['methods'][m]
                    for key,stored,source in [('jsd','mean_oracle_jsd',metric_rows),('topk','mean_topk',metric_rows),
                                              ('deletion_auc','mean_deletion_auc',functional_rows),('insertion_auc','mean_insertion_auc',functional_rows)]:
                        exact=float(np.mean([float(rec[f'{m}_{key}']) for rec in source]))
                        if not np.isclose(exact,float(z[stored]),atol=1e-10,rtol=0):
                            raise ValueError(f'Metric does not recompute: {name} seed {seed} {m} {key}')
                    rows.append({'dataset':j['dataset'],'scenario':name,'seed':seed,'method':m,
                        'accuracy':j['accuracy_full10k'],'jsd':z['mean_oracle_jsd'],
                        'topk':z['mean_topk'],'edi_proxy':z['heldout_predclass_summary_disagreement'],
                        'deletion':z['mean_deletion_auc'],'insertion':z['mean_insertion_auc'],
                        'bytes_per_client':float(np.mean(j['communication']['apgf_v6_generalized']['per_client_total_bytes'])) if m=='apgf_v6_gate_36fit_12val' else (j['communication']['qfscad_mean_bytes_per_client'] if m=='qfscad_v7' else (
                          0 if m in ('private_conv_48','private_field_48','input_only') else (
                          np.mean(j['communication']['static_fields']['static_total_bytes_per_client']) if m in ('fixed_fed_field_48','global_field_48','private_conv_with_fed_prior_48') else float('nan')) ))})
            elif tr.exists() and json.loads(tr.read_text()).get('status')=='FAIL_TASK_GATE':fail.append({'scenario':name,'seed':seed})
            else:missing.append({'scenario':name,'seed':seed})
    if rows:
        with (out/'per_seed_methods.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    comparisons=[]
    for name,seeds in cfg['seed_matrix'].items():
        r=[a for a in rows if a['scenario']==name]
        met={m:{z['seed']:z for z in r if z['method']==m} for m in METHODS}
        for baseline in ['private_field_48','fixed_fed_field_48','private_conv_48','private_conv_with_fed_prior_48','federated_conv_without_prior','input_only']:
            ids=sorted(set(met['qfscad_v7']) & set(met[baseline]));n=len(ids)
            if not ids:continue
            d=np.array([met[baseline][s]['jsd']-met['qfscad_v7'][s]['jsd'] for s in ids])
            td=np.array([met['qfscad_v7'][s]['topk']-met[baseline][s]['topk'] for s in ids])
            ed=np.array([met['qfscad_v7'][s]['edi_proxy']-met[baseline][s]['edi_proxy'] for s in ids])
            ci=stats.t.interval(.95,n-1,loc=d.mean(),scale=stats.sem(d)) if n>1 and np.std(d,ddof=1)>0 else (float('nan'),float('nan'))
            comparisons.append({'scenario':name,'baseline':baseline,'qualified_seeds':n,
                'planned_seeds':len(seeds),'mean_jsd_advantage':float(d.mean()),'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),
                'mean_topk_advantage':float(td.mean()),'change_in_disagreement_positive_is_worse':float(ed.mean()),
                'n_wins_jsd':int(sum(d>0)),
                'passes_min_superiority_gate':bool(n==len(seeds) and float(ci[0])>=cfg['primary_min_gain_over_private_conv']) if baseline=='private_conv_48' else None})
    if comparisons:
        with (out/'paired_seed_comparisons.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    # Strict prospective decision. No cherry-picked subsets or pooled-pixel pseudo-replication.
    advance_conditions=[]
    for name,seeds in cfg['seed_matrix'].items():
        byname=[r for r in rows if r['scenario']==name]
        bymethod={m:{r['seed']:r for r in byname if r['method']==m} for m in METHODS}
        per={'scenario':name,'all_predeclared_seeds_qualified':len(bymethod['qfscad_v7'])==len(seeds)}
        if per['all_predeclared_seeds_qualified']:
            ids=sorted(seeds)
            gains=[]
            for base in ('private_conv_48','private_conv_with_fed_prior_48'):
                dif=np.asarray([bymethod[base][seed]['jsd']-bymethod['qfscad_v7'][seed]['jsd'] for seed in ids]);
                ci=stats.t.interval(.95,len(ids)-1,loc=float(dif.mean()),scale=stats.sem(dif))
                per['gain_vs_'+base]=float(dif.mean());per['lower_ci_vs_'+base]=float(ci[0]);
                gains.append(bool(ci[0]>=cfg['primary_min_gain_over_private_conv']))
            base='fixed_fed_field_48'
            per['no_summary_drift_regression']=bool(np.mean([bymethod['qfscad_v7'][i]['edi_proxy']-bymethod[base][i]['edi_proxy'] for i in ids])<=0)
            per['no_deletion_regression']=bool(np.mean([bymethod['qfscad_v7'][i]['deletion']-bymethod[base][i]['deletion'] for i in ids])<=0)
            per['no_insertion_regression']=bool(np.mean([bymethod['qfscad_v7'][i]['insertion']-bymethod[base][i]['insertion'] for i in ids])>=0)
            per['advance_this_scenario']=bool(all(gains) and per['no_summary_drift_regression'] and per['no_deletion_regression'] and per['no_insertion_regression'])
        else:per['advance_this_scenario']=False
        advance_conditions.append(per)
    summary={'planned_training_runs':sum(map(len,cfg['seed_matrix'].values())),
        'promotion_gate_by_scenario':advance_conditions,
        'scientific_promotion':'ADVANCE_TO_EXTERNAL_REPLICATION' if all(x['advance_this_scenario'] for x in advance_conditions) and not missing and not fail else 'NO_ADVANCE_OR_INCOMPLETE',
        'complete_qualified_runs':len(set((r['scenario'],r['seed']) for r in rows)),
        'failed_task_gate':fail,'missing_or_incomplete':missing,'paired_comparisons':comparisons,
        'status':'COMPLETE' if not missing and not fail else 'INCOMPLETE_OR_TASK_GATE_FAILURES',
        'warnings':['Evaluation is conditional on CNN accuracy; missing/failed seeds cannot be silently replaced.',
                    'Nonofficial FedAttr class-template proxy is NOT an official xFedAlign reproduction.',
                    'Do not treat client/evaluation records as independent training seeds.',
                    'QF-SCAD research advancement additionally requires non-regression in drift and functional fidelity.']}
    (out/'aggregated_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['planned_training_runs','complete_qualified_runs','status']},indent=2))

if __name__=='__main__':main()
