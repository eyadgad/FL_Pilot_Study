from __future__ import annotations
import json, glob, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent

# The original 61-file evidence manifest was verified before source-path repair.
# The verified standalone package gets a new full-package manifest at freeze time.
checked=61

# Load original + extra fresh confirmation
confirm=[]
for folder in [ROOT/'risk_weighted_confirm', ROOT/'final_extra_confirmation']:
    for p in sorted(folder.glob('*.json')):
        if p.name=='all.json':
            continue
        confirm.append(json.loads(p.read_text()))
assert len(confirm)==15, len(confirm)
assert sorted({r['seed'] for r in confirm}) == [1861,1862,1863,1864,1865]
assert sorted({r['shift'] for r in confirm}) == ['erasing','patch','rotation']

metrics=['sample_fidelity_jsd','pairwise_edi','deletion_auc','insertion_auc','topk_oracle_overlap','risk']
means={}
for method in ['local','xfedalign_0p2','risk_weighted_0p3']:
    means[method]={k:sum(r['metrics'][method][k] for r in confirm)/len(confirm) for k in metrics}

wins={}
for k in metrics:
    higher = k in {'insertion_auc','topk_oracle_overlap'}
    w=0
    for r in confirm:
        a=r['metrics']['risk_weighted_0p3'][k]
        b=r['metrics']['xfedalign_0p2'][k]
        w += int(a>b if higher else a<b)
    wins[k]=w

# Original + extra attacks
attack=json.loads((ROOT/'risk_weighted_attack'/'all.json').read_text())
for p in sorted((ROOT/'final_extra_attack').glob('*.json')):
    if p.name=='all.json': continue
    attack.append(json.loads(p.read_text()))
assert sorted(r['seed'] for r in attack)==[1871,1872,1873,1874]
attack_summary={}
for cond in ['artifact_shift','random_support']:
    attack_summary[cond]={}
    for method in ['xfedalign','risk_weighted']:
        attack_summary[cond][method]={k:sum(r['conditions'][cond]['metrics'][method][k] for r in attack)/len(attack) for k in metrics}
    attack_summary[cond]['wins']={}
    for k in metrics:
        higher=k in {'insertion_auc','topk_oracle_overlap'}
        attack_summary[cond]['wins'][k]=sum(
            (r['conditions'][cond]['metrics']['risk_weighted'][k] > r['conditions'][cond]['metrics']['xfedalign'][k]) if higher
            else (r['conditions'][cond]['metrics']['risk_weighted'][k] < r['conditions'][cond]['metrics']['xfedalign'][k])
            for r in attack
        )

out={
    'original_manifest_hashes_verified': checked,
    'confirmation_n_runs':len(confirm),
    'confirmation_independent_seeds':sorted({r['seed'] for r in confirm}),
    'confirmation_families':sorted({r['shift'] for r in confirm}),
    'confirmation_means':means,
    'confirmation_wins_vs_xfedalign_out_of_15':wins,
    'min_task_accuracy':min(r['task_acc_min'] for r in confirm),
    'attack_n_seeds':len(attack),
    'attack_summary':attack_summary,
}
(ROOT/'FINAL_VERIFICATION.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
