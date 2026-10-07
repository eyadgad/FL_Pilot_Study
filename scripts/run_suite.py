#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys, hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def cfg_hash(path:Path):
    from ucpa_fl.config import load_config
    cfg=load_config(path).to_dict(); blob=json.dumps(cfg,sort_keys=True,default=str).encode()
    return hashlib.sha256(blob).hexdigest(),cfg

def completed_seeds(output_root, experiment, h):
    d=Path(output_root)/experiment; out=set()
    if not d.exists(): return out
    for man in d.glob('*/manifest.json'):
        try:
            m=json.loads(man.read_text())
            if m.get('status')=='completed' and m.get('config_sha256')==h and (man.parent/'final_results.json').exists(): out.add(int(m['seed']))
        except Exception: pass
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--tier',choices=['P','A','B','C','S','all'],default='A');p.add_argument('--manifest',default='configs/suite_manifest.yaml');p.add_argument('--no-resume',action='store_true');a=p.parse_args()
    man=yaml.safe_load((ROOT/a.manifest).read_text()); ledger=ROOT/'outputs'/'suite_ledger.jsonl'; ledger.parent.mkdir(exist_ok=True)
    selected=[x for x in man['configs'] if a.tier=='all' or x['tier']==a.tier]
    for item in selected:
        cpath=ROOT/'configs'/item['config']; h,cfg=cfg_hash(cpath); outroot=ROOT/cfg['logging']['output_root']; done=completed_seeds(outroot,item['experiment_name'],h)
        requested=set(map(int,cfg['seeds']))
        todo=sorted(requested if a.no_resume else requested-done)
        for sd in todo:
            proc=subprocess.run([sys.executable,str(ROOT/'run_experiment.py'),'--config',str(cpath),'--seed',str(sd)],cwd=ROOT)
            rc=proc.returncode
            with ledger.open('a') as f:
                f.write(json.dumps({'config':item['config'],'seed':sd,'status':'completed' if rc==0 else 'failed','config_sha256':h})+'\n')
            if rc: raise SystemExit(rc)
        print(f"{item['experiment_name']}: {len(todo)} new seeds; {len(requested)-len(todo)} skipped",flush=True)
    print(f'Finished tier {a.tier}: {len(selected)} configs')
if __name__=='__main__':main()
