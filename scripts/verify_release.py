#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'FILE_MANIFEST_SHA256.json').read_text());bad=[]
for rel,h in manifest.items():
    f=ROOT/rel
    if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:bad.append(rel)
if bad:raise SystemExit('FAILED: '+', '.join(bad))
print('PASS',len(manifest),'code/docs/config file hashes')
