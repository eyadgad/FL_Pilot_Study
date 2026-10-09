#!/usr/bin/env python3
"""Validate an extracted source-only research prototype. Unit integrity, NOT research proof."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run():
    manifest=ROOT/'MANIFEST.sha256'
    if not manifest.is_file():raise FileNotFoundError('Package checksum manifest missing')
    lines=[line.split('  ',1) for line in manifest.read_text().splitlines() if line.strip()]
    assert len(lines)==len(set(name for digest,name in lines)), 'Duplicate paths'
    actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    expected={name for digest,name in lines}
    if actual!=expected:raise AssertionError(f'Unexpected/missing files {sorted(actual^expected)}')
    for want,name in lines:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=want:
            raise AssertionError('Checksum mismatch: '+name)
    for x in actual:
        low=x.lower()
        if '/raw/' in low or low.endswith(('.pt','.pth','.npy','.npz','.zip','.tar.gz')):
            raise AssertionError('Data/checkpoint/archive in implementation-only package: '+x)
    freeze=json.loads((ROOT/'configs'/'APGF_FROZEN_YAML_HASHES.json').read_text())
    for fname,want in freeze.items():
        if hashlib.sha256((ROOT/'configs'/fname).read_bytes()).hexdigest()!=want:
            raise AssertionError('Prospective config mutated: '+fname)
    outcome={'status':'PASS_SOFTWARE_INTEGRITY_ONLY','verified_files':len(lines),'frozen_configs':len(freeze),'data_included':False,'checkpoint_included':False}
    print(json.dumps(outcome,indent=2));return outcome
if __name__=='__main__':run()
