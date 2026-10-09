#!/usr/bin/env python3
"""Real MNIST smoke of the exact generalized study methods from IG caches
produced previously from genuine MNIST and verified full-test CNN checkpoints.
No synthetic tensors are generated or treated as image data. Not a new CNN seed.
"""
import argparse,sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from study_v7.data import verify_data
from study_v7.codec import fit_fields
from study_v7.correction import train,train_private,predict
from study_v7.baselines import static_map
from study_v7.metrics import jsd,topk


def smoke(cache,root,out):
    ds=verify_data('mnist',root)
    z=np.load(cache,allow_pickle=False)
    if str(z['dataset_source_sha256'])!='14314248f0ada41a7d892e2b98b00ab037a6967d634361b0b7a9d61be9ceb342':
        raise ValueError('Not known genuine MNIST provenance')
    if float(z['full_test_accuracy'])<.95:raise ValueError('CNN accuracy failed')
    clients=[{k:z[f'c{i}_{k}'] for k in ('tx','ty','ex','ey')} for i in range(3)]
    if any(any(len(c[k])!=48 for k in c) for c in clients):raise ValueError('Real MNIST record count mismatch')
    xx=[c['tx'] for c in clients];yy=[c['ty'] for c in clients];
    fields,ledger=fit_fields(xx,yy,(28,28),peer_weight=.2)
    net,w,history=train(xx,yy,fields['fixed'],(28,28),123,rounds=8,steps_per_round=30)
    local=train_private(xx,yy,fields['private'],(28,28),123,steps=240)
    tab={}
    for name in ['input_only','private_field48','fixed_field48','private_conv48','qfscad_v7']:
        js=[];ov=[]
        for i,c in enumerate(clients):
            if name=='input_only':maps=static_map(c['ex'],np.ones(784))
            elif name=='private_field48':maps=static_map(c['ex'],fields['private'][i])
            elif name=='fixed_field48':maps=static_map(c['ex'],fields['fixed'][i])
            elif name=='private_conv48':maps=predict(local[i],c['ex'],fields['private'][i],(28,28))
            else:maps=predict(net,c['ex'],fields['fixed'][i],(28,28))
            js.extend([jsd(a,b) for a,b in zip(maps,c['ey'])]);ov.extend([topk(a,b,48) for a,b in zip(maps,c['ey'])])
        tab[name]={'mean_jsd':float(np.mean(js)),'top48':float(np.mean(ov))}
    result={'status':'PASS_REAL_MNIST_CACHED_IG_SMOKE','cache':str(cache),'accuracy_full10k':float(z['full_test_accuracy']),
            'teacher_count_per_client':48,'heldout_count_per_client':48,'methods':tab,
            'mean_wire_bytes_per_client':float(np.mean(np.array(ledger['static_total_bytes_per_client'])+np.array(w['model_total_bytes']))),
            'stage':'historical development checkpoint, NOT independent 60-run prospective confirmation',
            'dataset_file_sha256':ds['file_sha256'],'parameter_count':w['parameter_count']}
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'methods':tab,'wire_bytes':result['mean_wire_bytes_per_client']},indent=2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--data-root',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();smoke(args.cache,args.data_root,args.out)
