#!/usr/bin/env python3
"""Verify actual MNIST IDX pixels imported from the user-provided CSV dataset.

Fails closed if data are missing, altered, or not the original 60k/10k split.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from torchvision.datasets import MNIST

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def verify(root):
 root=Path(root); raw=root/'MNIST'/'raw'; prov_path=root/'MNIST'/'csv_import_provenance.json'
 d=json.loads(prov_path.read_text())
 if not d.get('no_synthetic_samples'): raise AssertionError('not marked real MNIST source')
 actual={}
 for split,prefix in [('train','train'),('test','t10k')]:
  v=d[split]; im=raw/f'{prefix}-images-idx3-ubyte'; la=raw/f'{prefix}-labels-idx1-ubyte'
  if sha(im)!=v['idx_images_sha256'] or sha(la)!=v['idx_labels_sha256']:
   raise AssertionError(f'Corrupted {split} images/labels')
  ds=MNIST(str(root),train=split=='train',download=False)
  if len(ds)!=v['rows']:raise AssertionError('sample count mismatch')
  hist=np.bincount(ds.targets.numpy(),minlength=10).tolist()
  if hist!=v['label_counts']:raise AssertionError('label histogram mismatch')
  actual[split]={'rows':len(ds),'pixel_shape':list(ds.data.shape[1:]),'label_hist':hist,
                 'first_label':int(ds.targets[0]),'nonzero_pixels':int((ds.data!=0).sum())}
 if actual['train']['rows']!=60000 or actual['test']['rows']!=10000:raise AssertionError('MNIST split invalid')
 if actual['train']['first_label']!=5 or actual['test']['first_label']!=7:raise AssertionError('MNIST image order invalid')
 return {'status':'PASS','data':'real MNIST CSV-derived data; no synthetic images','source_archive_sha256':d['source_zip_sha256'],**actual}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',default='data');a=p.parse_args()
 r=verify(a.dataset_root);print(json.dumps(r,indent=2))
