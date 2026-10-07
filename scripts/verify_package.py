#!/usr/bin/env python3
from pathlib import Path
import subprocess,sys,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
checks=[]
def run(name,cmd):
    p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);checks.append((name,p.returncode,p.stdout[-1200:],p.stderr[-1200:]));return p.returncode
rc=run('compile',[sys.executable,'-m','compileall','-q','ucpa_fl','scripts','run_experiment.py','aggregate_results.py','evaluate_cfba_gates.py'])
rc|=run('pytest',[sys.executable,'-m','pytest','-q'])
# Parse every config, including generated suite.
from ucpa_fl.config import load_config
for p in sorted((ROOT/'configs').glob('*.yaml')):
    if p.name.startswith('_'):continue
    try:load_config(p) if p.name!='suite_manifest.yaml' else yaml.safe_load(p.read_text())
    except Exception as e:print(f'CONFIG FAIL {p}: {e}');rc=1
for name,code,out,err in checks:
    print(f'[{"PASS" if code==0 else "FAIL"}] {name}\n{out}\n{err}')
raise SystemExit(1 if rc else 0)
