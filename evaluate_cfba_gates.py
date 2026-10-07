#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np

CORE=[
    'core_mnist_rotation','core_mnist_patch',
    'core_cifar10_rotation','core_cifar10_color',
]
ATTACKS=['robust_mnist_rotation_artifact_shift','robust_mnist_rotation_random_support']
EXPECTED_SEEDS={1701,1702,1703,1704,1705}


def f(x):
    try:return float(x)
    except:return float('nan')

def load_rows(path):
    with open(path,newline='',encoding='utf-8') as fh:
        out=[]
        for r in csv.DictReader(fh):
            r=dict(r);r['seed']=int(r['seed'])
            for k,v in list(r.items()):
                if k not in ('experiment','scenario','method','run_dir','config_sha256','seed'):
                    r[k]=f(v)
            out.append(r)
        return out

def lookup(rows,exp,scenario,method):
    return {r['seed']:r for r in rows if r['experiment']==exp and r['scenario']==scenario and r['method']==method}

def mean(vals):
    a=np.asarray([x for x in vals if np.isfinite(x)],float)
    return float(a.mean()) if len(a) else float('nan')

def rel_improve_lower(a,b):
    # a is candidate, b comparator
    if not (np.isfinite(a) and np.isfinite(b)) or abs(b)<1e-15:return float('nan')
    return float((b-a)/abs(b))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    rows=load_rows(args.runs)
    report={'method':'CFBA-CRC','frozen_alpha':0.05,'expected_seeds':sorted(EXPECTED_SEEDS),'gates':{},'family_details':{}}

    # Completeness first. We deliberately return PENDING instead of evaluating a partial study.
    required=[]
    for exp in CORE:
        for m in ('local','xfedalign_median','global_crc','cfba_crc'):
            required.append((exp,'clean',m))
    for exp in ATTACKS:
        for sc in ('clean','attacked'):
            for m in ('local','xfedalign_median','global_crc','cfba_crc'):
                required.append((exp,sc,m))
    missing=[]
    for key in required:
        got=set(lookup(rows,*key))
        if got!=EXPECTED_SEEDS: missing.append({'key':list(key),'got':sorted(got),'missing':sorted(EXPECTED_SEEDS-got)})
    if missing:
        report['decision']='PENDING';report['missing_evidence']=missing
        Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(report,indent=2,sort_keys=True))
        print('PENDING');return

    # G1: Task validity. Explanations are uninterpretable if task learning failed.
    task_ok=True; task_details={}
    for exp in CORE:
        r=lookup(rows,exp,'clean','local')
        vals=[r[s]['client_task_accuracy_mean'] for s in sorted(EXPECTED_SEEDS)]
        threshold=.80 if 'mnist' in exp else .35
        ok=mean(vals)>=threshold and all(np.isfinite(vals))
        task_details[exp]={'mean_client_accuracy':mean(vals),'threshold':threshold,'pass':bool(ok)};task_ok &= ok
    report['gates']['G1_task_validity']={'pass':bool(task_ok),'details':task_details}

    # G2: Every core client must have enough calibration data to certify beta=0 at least.
    cert_ok=True; cert_details={}
    for exp in CORE:
        r=lookup(rows,exp,'clean','cfba_crc'); vals=[r[s].get('certified_fraction',float('nan')) for s in sorted(EXPECTED_SEEDS)]
        ok=all(np.isfinite(vals)) and min(vals)>=1.0-1e-12
        cert_details[exp]={'fractions':vals,'pass':bool(ok)}; cert_ok &= ok
    report['gates']['G2_certification_completeness']={'pass':bool(cert_ok),'details':cert_details}

    # G3: Held-out bounded composite risk must remain within the frozen risk budget on average in every core family.
    risk_ok=True;risk_details={}
    for exp in CORE:
        r=lookup(rows,exp,'clean','cfba_crc'); vals=[r[s]['excess_fidelity_risk'] for s in sorted(EXPECTED_SEEDS)]
        m=mean(vals);ok=np.isfinite(m) and m<=.05+1e-12
        risk_details[exp]={'per_seed':vals,'mean':m,'alpha':.05,'pass':bool(ok)};risk_ok &= ok
    report['gates']['G3_heldout_composite_risk']={'pass':bool(risk_ok),'details':risk_details}

    # G4: Client-specific certification must add value over one globally safe beta.
    adapt_passes=0; adapt_details={}
    for exp in CORE:
        a=lookup(rows,exp,'clean','cfba_crc'); b=lookup(rows,exp,'clean','global_crc')
        av=[a[s]['pairwise_edi'] for s in sorted(EXPECTED_SEEDS)]; bv=[b[s]['pairwise_edi'] for s in sorted(EXPECTED_SEEDS)]
        ma,mb=mean(av),mean(bv); wins=sum(1 for x,y in zip(av,bv) if x<y-1e-12); rel=rel_improve_lower(ma,mb)
        ok=np.isfinite(rel) and rel>=.05 and wins>=4
        adapt_passes+=int(ok); adapt_details[exp]={'cfba_mean':ma,'global_crc_mean':mb,'relative_improvement':rel,'wins':wins,'n':5,'pass':bool(ok)}
    adapt_ok=adapt_passes>=3
    report['gates']['G4_client_specific_value']={'pass':bool(adapt_ok),'families_passing':adapt_passes,'required':3,'details':adapt_details}

    # G5: Pooled non-inferiority to Local-XAI on each fidelity measure.
    pairs=[]
    for exp in CORE:
        a=lookup(rows,exp,'clean','cfba_crc'); b=lookup(rows,exp,'clean','local')
        for s in sorted(EXPECTED_SEEDS):pairs.append((a[s],b[s]))
    deltas={
        'sample_fidelity_jsd':mean([a['sample_fidelity_jsd']-b['sample_fidelity_jsd'] for a,b in pairs]),
        'deletion_auc':mean([a['deletion_auc']-b['deletion_auc'] for a,b in pairs]),
        'insertion_auc':mean([a['insertion_auc']-b['insertion_auc'] for a,b in pairs]),
        'topk_oracle_overlap':mean([a['topk_oracle_overlap']-b['topk_oracle_overlap'] for a,b in pairs]),
    }
    limits={'sample_fidelity_jsd':.02,'deletion_auc':.01,'insertion_auc':-.01,'topk_oracle_overlap':-.05}
    checks={
        'sample_fidelity_jsd':deltas['sample_fidelity_jsd']<=.02,
        'deletion_auc':deltas['deletion_auc']<=.01,
        'insertion_auc':deltas['insertion_auc']>=-.01,
        'topk_oracle_overlap':deltas['topk_oracle_overlap']>=-.05,
    }
    fid_ok=all(checks.values())
    report['gates']['G5_fidelity_noninferiority_to_local']={'pass':bool(fid_ok),'pooled_cfba_minus_local':deltas,'limits':limits,'checks':{k:bool(v) for k,v in checks.items()}}

    # G6: CFBA adds no server artifact payload relative to xFedAlign mean/support artifacts.
    comm_ok=True;comm_details={}
    for exp in CORE:
        a=lookup(rows,exp,'clean','cfba_crc'); b=lookup(rows,exp,'clean','xfedalign_median')
        dif=[abs(a[s]['communication_bytes_per_client_artifact']-b[s]['communication_bytes_per_client_artifact']) for s in sorted(EXPECTED_SEEDS)]
        ok=max(dif)<=1e-9;comm_ok &= ok;comm_details[exp]={'max_abs_byte_difference':max(dif),'pass':bool(ok)}
    report['gates']['G6_communication_parity']={'pass':bool(comm_ok),'details':comm_details}

    # G7: Recalibration against attacked priors must keep held-out risk inside budget and avoid becoming worse than Local on EDI/fidelity.
    attack_ok=True;attack_details={}
    for exp in ATTACKS:
        a=lookup(rows,exp,'attacked','cfba_crc'); l=lookup(rows,exp,'attacked','local')
        risks=[a[s]['excess_fidelity_risk'] for s in sorted(EXPECTED_SEEDS)]
        jsd=[a[s]['sample_fidelity_jsd']-l[s]['sample_fidelity_jsd'] for s in sorted(EXPECTED_SEEDS)]
        edi=[a[s]['pairwise_edi']-l[s]['pairwise_edi'] for s in sorted(EXPECTED_SEEDS)]
        cert=[a[s].get('certified_fraction',float('nan')) for s in sorted(EXPECTED_SEEDS)]
        ok=(mean(risks)<=.05+1e-12 and mean(jsd)<=.02 and mean(edi)<=1e-12 and min(cert)>=1-1e-12)
        attack_ok &= ok
        attack_details[exp]={'mean_risk':mean(risks),'mean_cfba_minus_local_sample_jsd':mean(jsd),'mean_cfba_minus_local_edi':mean(edi),'min_certified_fraction':min(cert),'pass':bool(ok)}
    report['gates']['G7_attack_recalibration_safety']={'pass':bool(attack_ok),'details':attack_details}

    all_pass=all(v['pass'] for v in report['gates'].values())
    report['decision']='CONTINUE' if all_pass else 'REVISE_OR_KILL'
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(report['decision'])

if __name__=='__main__':main()
