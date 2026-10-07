#!/usr/bin/env python3
"""Offline static verification, tests, prereg and frozen seed independence."""
from pathlib import Path
import sys,subprocess,json
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ucpa_fl.config import load_config

p=subprocess.run([sys.executable,'-m','compileall','-q','ucpa_fl','scripts','run_experiment.py','aggregate_results.py','evaluate_rwssa_gates.py'],cwd=ROOT)
assert p.returncode==0,'Compile failed'
p=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT)
assert p.returncode==0,'Unit tests failed'
manifest=yaml.safe_load((ROOT/'configs/suite_manifest.yaml').read_text())
allseeds={tier:set() for tier in ['P','A','B','C','S']}
checks=0
for item in manifest['configs']:
    cfg=load_config(ROOT/'configs'/item['config']);checks+=1
    assert cfg.experiment_name==item['experiment_name']
    assert cfg.tags['tier']==item['tier']
    assert 'rwssa' in cfg.alignment.methods and 'rwssa_sparse' in cfg.alignment.methods
    assert len(cfg.seeds)==len(set(cfg.seeds))
    if item['tier'] in ('P','A'):
        assert cfg.alignment.rwssa_rho==.30
        assert cfg.alignment.rwssa_risk_mode=='jsd_topk'
        assert cfg.alignment.rwssa_grid==[0,.25,.5,.75,1.]
    allseeds[item['tier']].update(cfg.seeds)
for a,b in [('P','A'),('P','B'),('A','B'),('P','S'),('A','S')]:
    assert not allseeds[a] & allseeds[b],f'Overlap between tiers {a}/{b}'
# Packaged verification outputs may include sanity seeds but not trial/confirmation.
for f in (ROOT/'verification').rglob('manifest.json'):
    d=json.loads(f.read_text())
    assert int(d['seed']) not in allseeds['P']|allseeds['A']
print(json.dumps({'status':'PASS','configs':checks,'seeds':{k:sorted(v) for k,v in allseeds.items()}},indent=2))
