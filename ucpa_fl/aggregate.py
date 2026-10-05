from __future__ import annotations
from pathlib import Path
import json, csv, math
import numpy as np
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon=None

METRICS=['artifact_fidelity_jsd','sample_fidelity_jsd','pairwise_edi','reference_edi','deletion_auc','insertion_auc','topk_oracle_overlap','communication_bytes_per_client_artifact']

def _ci(vals,seed=0,B=2000):
    vals=np.asarray(vals,float); vals=vals[np.isfinite(vals)]
    if len(vals)==0:return (np.nan,np.nan,np.nan,np.nan)
    mean=float(vals.mean()); sd=float(vals.std(ddof=1)) if len(vals)>1 else 0.
    if len(vals)==1:return mean,sd,mean,mean
    rng=np.random.default_rng(seed); boots=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(B)])
    return mean,sd,float(np.quantile(boots,.025)),float(np.quantile(boots,.975))

def collect(output_root):
    raw=[]
    for p in Path(output_root).rglob('final_results.json'):
        d=json.loads(p.read_text()); run=p.parent
        man=json.loads((run/'manifest.json').read_text())
        if man.get('status')!='completed': continue
        for scenario in ('clean','attacked'):
            if not d.get(scenario): continue
            for method,rec in d[scenario].items():
                row={'experiment':man['experiment'],'seed':d['seed'],'scenario':scenario,'method':method,'run_dir':str(run),
                     '_finished_unix':man.get('finished_unix',0.0),'config_sha256':man.get('config_sha256',''),
                     'task_accuracy':d.get('task',{}).get('accuracy',np.nan),'client_task_accuracy_mean':d.get('task',{}).get('client_accuracy_mean',np.nan),
                     'client_task_accuracy_worst':d.get('task',{}).get('client_accuracy_worst',np.nan),
                     'artifact_mia_mean_only_auc':d.get('privacy',{}).get('mean_only_auc',np.nan),
                     'artifact_mia_mean_variance_auc':d.get('privacy',{}).get('mean_variance_auc',np.nan)}
                row.update(rec); raw.append(row)
    # Never double-count reruns. Select the latest completed run per frozen experiment/seed/scenario/method,
    # but retain duplicate provenance for an audit file.
    latest={}; duplicates=[]
    for r in raw:
        key=(r['experiment'],r['seed'],r['scenario'],r['method'])
        if key in latest:
            duplicates.append({'key':list(key),'kept_or_replaced':[latest[key]['run_dir'],r['run_dir']]})
            if r['_finished_unix']>=latest[key]['_finished_unix']: latest[key]=r
        else: latest[key]=r
    rows=[]
    for r in latest.values():
        r=dict(r);r.pop('_finished_unix',None);rows.append(r)
    return rows,duplicates

def aggregate(output_root, out_dir):
    rows,duplicates=collect(output_root); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'duplicate_runs.json').write_text(json.dumps(duplicates,indent=2),encoding='utf-8')
    if not rows:return []
    cols=sorted(set().union(*(r.keys() for r in rows)))
    with open(out/'runs.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    groups={}
    for r in rows: groups.setdefault((r['experiment'],r['scenario'],r['method']),[]).append(r)
    summary=[]
    for (exp,sc,m),rs in sorted(groups.items()):
        rec={'experiment':exp,'scenario':sc,'method':m,'n_seeds':len({r['seed'] for r in rs})}
        for metric in METRICS:
            vals=[r.get(metric,np.nan) for r in rs]; mean,sd,lo,hi=_ci(vals)
            rec[f'{metric}_mean']=mean;rec[f'{metric}_sd']=sd;rec[f'{metric}_ci95_lo']=lo;rec[f'{metric}_ci95_hi']=hi
        summary.append(rec)
    cols=sorted(set().union(*(r.keys() for r in summary)))
    with open(out/'summary.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(summary)
    # Paired NC-SACPA-vs-xFedAlign differences, the primary comparison.
    paired=[]
    by={(r['experiment'],r['scenario'],r['seed'],r['method']):r for r in rows}
    exps=sorted(set((r['experiment'],r['scenario']) for r in rows))
    for exp,sc in exps:
        seeds=sorted(set(r['seed'] for r in rows if r['experiment']==exp and r['scenario']==sc))
        for metric in METRICS:
            dif=[]
            for s in seeds:
                a=by.get((exp,sc,s,'nc_sacpa')); b=by.get((exp,sc,s,'xfedalign_median'))
                if a and b and np.isfinite(a.get(metric,np.nan)) and np.isfinite(b.get(metric,np.nan)): dif.append(a[metric]-b[metric])
            if dif:
                mean,sd,lo,hi=_ci(dif,seed=33)
                wins=sum(1 for x in dif if x<0) if metric not in ('insertion_auc','topk_oracle_overlap') else sum(1 for x in dif if x>0)
                pval=float('nan')
                if wilcoxon is not None and len(dif)>=3 and not np.allclose(dif,0):
                    try: pval=float(wilcoxon(dif,alternative='two-sided',zero_method='wilcox').pvalue)
                    except Exception: pass
                paired.append({'experiment':exp,'scenario':sc,'metric':metric,'n':len(dif),'ncsacpa_minus_xfedalign_mean':mean,'sd':sd,'ci95_lo':lo,'ci95_hi':hi,'directional_wins':wins,'wilcoxon_two_sided_p':pval})
    if paired:
        cols=list(paired[0]);
        with open(out/'paired_ncsacpa_vs_xfedalign.csv','w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(paired)
    md=['# Aggregate results','',f'Runs found: {len(rows)}','', 'Primary paired comparison is NC-SACPA minus xFedAlign-style median prior. Negative is favorable for lower-is-better metrics (JSD/EDI/deletion); positive is favorable for insertion/overlap.','']
    for rec in summary:
        md.append(f"## {rec['experiment']} / {rec['scenario']} / {rec['method']} (n={rec['n_seeds']})")
        md.append(f"- artifact fidelity JSD: {rec['artifact_fidelity_jsd_mean']:.6f} [{rec['artifact_fidelity_jsd_ci95_lo']:.6f}, {rec['artifact_fidelity_jsd_ci95_hi']:.6f}]")
        md.append(f"- pairwise EDI: {rec['pairwise_edi_mean']:.6f}")
        md.append(f"- deletion / insertion AUC: {rec['deletion_auc_mean']:.6f} / {rec['insertion_auc_mean']:.6f}")
        md.append('')
    (out/'summary.md').write_text('\n'.join(md),encoding='utf-8')
    return summary
