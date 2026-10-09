"""Offline-only real benchmark validation and explanation record generation."""
from __future__ import annotations
import hashlib,struct,json,pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ucpa_fl.explain import integrated_gradients


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def verify_data(name,root):
    root=Path(root)
    if name=='mnist':
        p=root/'MNIST'/'raw'
        specs={'train-images-idx3-ubyte':(2051,60000,784),'train-labels-idx1-ubyte':(2049,60000,1),
               't10k-images-idx3-ubyte':(2051,10000,784),'t10k-labels-idx1-ubyte':(2049,10000,1)}
        hashes={}
        for file,(magic,n,record) in specs.items():
            f=p/file
            if not f.is_file():raise FileNotFoundError(f'Required ORIGINAL MNIST IDX file not found: {f}')
            with f.open('rb') as stream:
                hdr=stream.read(16 if magic==2051 else 8)
            found_magic,found_n=struct.unpack('>II',hdr[:8])
            if found_magic!=magic or found_n!=n:raise ValueError(f'Wrong MNIST header/count {file}')
            off=16 if magic==2051 else 8
            if magic==2051 and struct.unpack('>II',hdr[8:16])!=(28,28):raise ValueError('Wrong MNIST shape')
            if f.stat().st_size!=off+n*record:raise ValueError('Wrong IDX file length')
            if magic==2049 and not set(np.fromfile(f,dtype='u1',offset=8)).issubset(set(range(10))):
                raise ValueError('Invalid MNIST class labels')
            hashes[file]=sha(f)
        expected_path=Path(__file__).resolve().parents[1]/'configs'/'EXPECTED_ORIGINAL_MNIST_IDX_SHA256.json'
        if not expected_path.is_file():raise ValueError('Missing pinned genuine MNIST SHA256 reference')
        expected=json.loads(expected_path.read_text())
        if hashes!=expected:raise ValueError('MNIST IDX bytes differ from user-provided verified genuine dataset')
        return {'dataset':name,'train_count':60000,'test_count':10000,'sample_shape':[1,28,28],
                'file_sha256':hashes,'source':'local original MNIST IDX; no download'}
    if name=='cifar10':
        p=root/'cifar-10-batches-py'
        files=[f'data_batch_{j}' for j in range(1,6)]+['test_batch']
        if not (p/'batches.meta').is_file():raise FileNotFoundError('Original CIFAR-10 batches.meta is required')
        hashes={}
        for name_ in files:
            f=p/name_
            if not f.is_file():raise FileNotFoundError(f'Original CIFAR-10 batch unavailable: {f} (no download fallback)')
            # Each original CIFAR-10 pickle file is 10,000 3x32x32 uint8 records.
            with f.open('rb') as fd:obj=pickle.load(fd,encoding='bytes')
            x=obj[b'data'];y=obj.get(b'labels',obj.get('labels'))
            if x.shape!=(10000,3072) or x.dtype!=np.uint8 or len(y)!=10000:
                raise ValueError(f'Invalid CIFAR-10 batch shape/count: {name_}')
            if min(y)<0 or max(y)>9:raise ValueError('Invalid CIFAR-10 labels')
            hashes[name_]=sha(f)
        hashes['batches.meta']=sha(p/'batches.meta')
        return {'dataset':name,'train_count':50000,'test_count':10000,'sample_shape':[3,32,32],
                'file_sha256':hashes,'source':'local original CIFAR-10 python batches; no download'}
    raise ValueError('Only genuine MNIST and CIFAR-10 are permitted')


def collect(model,dataset,count,device,steps,spatial_hw,batch_size=12):
    """Spatial attribution: per-channel absolute IG summed for RGB, not flatten 3072.
    Input feature is per-pixel mean RGB intensity, one channel (same 77 params).
    The predicted task class is frozen-model argmax, not ground-truth class.
    """
    xx=[];yy=[];pp=[]
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    model.eval()
    for X,_ in loader:
        if len(pp)>=count:break
        X=X.to(device)
        with torch.no_grad():pred=model(X).argmax(1)
        oracle=integrated_gradients(model,X,pred,steps=steps).abs().sum(dim=1).flatten(1)
        oracle=oracle/(oracle.sum(dim=1,keepdim=True)+1e-12)
        x=X.mean(dim=1).flatten(1)
        take=min(len(pred),count-len(pp))
        xx.extend(x[:take].detach().cpu().numpy())
        yy.extend(oracle[:take].detach().cpu().numpy())
        pp.extend(pred[:take].detach().cpu().numpy().tolist())
    if len(pp)!=count:raise ValueError(f'Insufficient genuine data: required {count} but got {len(pp)}')
    return {'x':np.asarray(xx,dtype=np.float32), 'ig':np.asarray(yy,dtype=np.float32),
            'pred':np.asarray(pp,dtype=np.int64)}


def save_splits(bundle,out):
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    arrays={}
    for cid in range(len(bundle.task_clients)):
        for name,group in [('task',bundle.task_clients),('surrogate',bundle.surrogate_clients),
                           ('artifact',bundle.artifact_clients),('teacher',bundle.calibration_clients),
                           ('evaluation',bundle.eval_clients)]:
            arrays[f'c{cid}_{name}']=np.asarray(group[cid].indices,dtype=np.int64)
    joined=np.concatenate(list(arrays.values()))
    if len(set(joined.tolist()))!=len(joined):raise ValueError('Overlapping partition indexes')
    np.savez_compressed(p,**arrays)
    return sha(p), {k:int(len(v)) for k,v in arrays.items()}
