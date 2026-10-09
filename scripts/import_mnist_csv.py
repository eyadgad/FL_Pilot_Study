#!/usr/bin/env python3
"""Import REAL MNIST CSV shards from user's project.zip into torchvision IDX cache.

The CSV format is label,pixel0,...,pixel783, with 0-255 integer pixels.
No synthetic data are generated, and source test records remain disjoint.
"""
from __future__ import annotations
import argparse, hashlib, json, struct, zipfile
from pathlib import Path
import numpy as np

TRAIN = [f'data/mnist/MNIST_train_{j}.csv' for j in range(1, 7)]
TEST = ['data/mnist/MNIST_test.csv']
HEADER = 'label,'+','.join(f'pixel{j}' for j in range(784))

def convert(zf, paths, image_path: Path, label_path: Path):
    n=0; label_hist=[0]*10; file_digests={}; buffers=[]
    image_path.parent.mkdir(parents=True,exist_ok=True)
    with image_path.open('wb') as io, label_path.open('wb') as lo:
        io.write(struct.pack('>IIII',2051,0,28,28))
        lo.write(struct.pack('>II',2049,0))
        for path in paths:
            h=hashlib.sha256(); count=0
            with zf.open(path) as stream:
                first=stream.readline(); h.update(first)
                if first.decode('ascii').strip()!=HEADER:
                    raise ValueError(f'Unexpected header: {path}')
                for line in stream:
                    h.update(line)
                    row=np.fromstring(line.decode('ascii'), dtype=np.int16, sep=',')
                    if row.size!=785 or np.any(row<0) or np.any(row>255) or row[0]>9:
                        raise ValueError(f'Invalid MNIST row at {path}: row={count+1}')
                    label=int(row[0]); lo.write(bytes([label])); io.write(row[1:].astype(np.uint8).tobytes())
                    label_hist[label]+=1; count+=1; n+=1
            file_digests[path]={'rows':count,'sha256':h.hexdigest()}
        io.seek(4);io.write(struct.pack('>I',n))
        lo.seek(4);lo.write(struct.pack('>I',n))
    return {'rows':n,'label_counts':label_hist,'source_files':file_digests,
            'idx_images_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
            'idx_labels_sha256':hashlib.sha256(label_path.read_bytes()).hexdigest()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-zip',required=True);p.add_argument('--dataset-root',default='data');a=p.parse_args()
    root=Path(a.dataset_root)/'MNIST'/'raw';root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.source_zip) as z:
        for item in TRAIN+TEST:
            if item not in z.namelist():raise FileNotFoundError(item)
        train=convert(z,TRAIN,root/'train-images-idx3-ubyte',root/'train-labels-idx1-ubyte')
        test=convert(z,TEST,root/'t10k-images-idx3-ubyte',root/'t10k-labels-idx1-ubyte')
    report={'dataset':'MNIST, real handwritten digit pixels from supplied CSV shards',
            'train':train,'test':test,'source_archive':str(Path(a.source_zip).resolve()),
            'source_zip_sha256':hashlib.sha256(Path(a.source_zip).read_bytes()).hexdigest(),
            'no_synthetic_samples':True}
    out=Path(a.dataset_root)/'MNIST'/'csv_import_provenance.json'
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps({'train_rows':train['rows'],'test_rows':test['rows'],'train_hist':train['label_counts'], 'test_hist':test['label_counts'],'manifest':str(out)},indent=2))
if __name__=='__main__':main()
