#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json, sys
p=argparse.ArgumentParser();p.add_argument('--outputs',default='./outputs');a=p.parse_args()
root=Path(a.outputs); bad=[]; good=0
for integ in root.rglob('integrity.sha256'):
    run=integ.parent; ok=True
    for line in integ.read_text().splitlines():
        digest,rel=line.split('  ',1); q=run/rel
        if not q.exists() or hashlib.sha256(q.read_bytes()).hexdigest()!=digest:
            bad.append(f'{run}: {rel}');ok=False
    if ok:good+=1
print(json.dumps({'verified_runs':good,'failures':bad},indent=2))
raise SystemExit(1 if bad else 0)
