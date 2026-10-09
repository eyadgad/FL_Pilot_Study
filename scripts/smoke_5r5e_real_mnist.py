#!/usr/bin/env python3
"""Genuine-MNIST *engineering* smoke: exactly 5 FedAvg rounds x 5 local epochs.
Reduced-scale real-image samples for execution safety; NOT scientific validation.
No synthetic images, data downloading or hidden checkpoint fallback.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from torch.utils.data import Subset,DataLoader
from torchvision import datasets,transforms
from torch import nn
from torch.nn import functional as F

class MNISTSmokeCNN(nn.Module):
    """Reduced two-convolution model ONLY for real-image engineering smoke."""
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,8,3,padding=1)
        self.conv2=nn.Conv2d(8,16,3,padding=1)
        self.fc1=nn.Linear(16*7*7,64)
        self.fc2=nn.Linear(64,10)
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),2)
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))
from ucpa_fl.repro import set_seed,resolve_device
from ucpa_fl.federated import train_local,_weighted_average,evaluate
from study_v7.data import verify_data,collect
from study_v7.codec import fit_fields
from study_v7.correction import train,train_private,predict
from study_v7.metrics import jsd,topk
from study_v7.published_baselines import collect_published_controls

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',default='data');p.add_argument('--out',default='engineering_smoke_real_mnist')
    p.add_argument('--device',default='auto');p.add_argument('--threads',type=int,default=3)
    p.add_argument('--train-per-client',type=int,default=650)
    p.add_argument('--test-gate',type=float,default=.75,help='ENGINEERING ONLY; full-study floor remains .95')
    a=p.parse_args()
    torch.set_num_threads(a.threads);set_seed(20261009,True)
    device=resolve_device(a.device)
    data_root=Path(a.data_root).resolve(); out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    prov=verify_data('mnist',data_root)
    ds=datasets.MNIST(str(data_root),train=True,download=False,transform=transforms.ToTensor())
    test=datasets.MNIST(str(data_root),train=False,download=False,transform=transforms.ToTensor())
    rng=np.random.default_rng(20261009)
    labels=np.asarray(ds.targets)
    train_ids=[]
    # Three IID-like balanced clients, every image is a genuine distinct original MNIST image.
    for c in range(10):
        z=np.where(labels==c)[0];rng.shuffle(z);train_ids.extend(z[:3*a.train_per_client//10])
    train_ids=np.asarray(train_ids,dtype=np.int64);rng.shuffle(train_ids)
    clients=[Subset(ds,z.tolist()) for z in np.array_split(train_ids,3)]
    available=np.setdiff1d(np.arange(60000),train_ids,assume_unique=True)
    rng.shuffle(available)
    teacher_sets=[];held_sets=[]
    for cid in range(3):
        indices=available[cid*80:(cid+1)*80]
        teacher_sets.append(Subset(ds,indices[:48].tolist()))
        held_sets.append(Subset(ds,indices[48:64].tolist()))
    model=MNISTSmokeCNN().to(device)
    history=[];start=time.perf_counter()
    for round_ in range(5):
        states=[];weights=[];losses=[]
        from copy import deepcopy
        for cid in range(3):
            local=deepcopy(model).to(device)
            loader=DataLoader(clients[cid],batch_size=96,shuffle=True,generator=torch.Generator().manual_seed(20261009+cid+round_*100))
            loss=train_local(local,loader,device,epochs=5,lr=.04,momentum=.9,weight_decay=.0001)
            states.append({k:v.detach().cpu() for k,v in local.state_dict().items()})
            weights.append(len(clients[cid]));losses.append(loss)
        model.load_state_dict(_weighted_average(states,weights))
        history.append({'round':round_+1,'local_epochs':5,'loss':float(np.mean(losses))})
        print(f'ROUND {round_+1}/5 local_epochs=5 train_loss={np.mean(losses):.5f}',flush=True)
    full=evaluate(model,DataLoader(test,batch_size=256,shuffle=False),device)
    report={'status':'TRAINED','real_dataset':prov,'training_rounds':5,'local_epochs':5,
      'train_per_client':list(map(len,clients)),'full10k_task_accuracy':full['accuracy'],
      'engineering_gate':a.test_gate,'elapsed_train_sec':round(time.perf_counter()-start,2),
      'train_rounds':history,'device':str(device),'smoke_model':'MNISTSmokeCNN 2 convolution blocks (full study remains unchanged)','scientific_claim':False,
      'published_controls':['gradient_saliency','gradcam'],'no_synthetic_data':True}
    if full['accuracy'] < a.test_gate:
        report['status']='FAIL_ENGINEERING_TASK_GATE';(out/'result.json').write_text(json.dumps(report,indent=2));
        print('FAIL_CNN_GATE',full['accuracy']);return
    # Real, completely held-out source-image records; shared task CNN.
    teacher=[collect(model,s,48,device,4,(28,28),batch_size=8) for s in teacher_sets]
    held=[collect(model,s,16,device,4,(28,28),batch_size=8) for s in held_sets]
    fields,ledger=fit_fields([t['x'] for t in teacher],[t['ig'] for t in teacher],(28,28))
    net,nledger,losses=train([t['x'] for t in teacher],[t['ig'] for t in teacher],fields['fixed'],(28,28),seed=9911,
            rounds=2,steps_per_round=8,device=device)
    private=train_private([t['x'] for t in teacher],[t['ig'] for t in teacher],fields['private'],(28,28),seed=9911,steps=16,device=device)
    m={'qfscad_v7':[],'private_conv_48':[],'fixed_field_48':[],'gradient_saliency':[],'gradcam':[]}
    from study_v7.baselines import static_map
    for i in range(3):
        controls=collect_published_controls(model,held_sets[i],16,device)
        preds={'qfscad_v7':predict(net,held[i]['x'],fields['fixed'][i],(28,28)),
         'private_conv_48':predict(private[i],held[i]['x'],fields['private'][i],(28,28)),
         'fixed_field_48':static_map(held[i]['x'],fields['fixed'][i],1.2),**controls}
        for method,values in preds.items():
            for y,z in zip(values,held[i]['ig']):
                m[method].append((jsd(y,z),topk(y,z,48)))
    report['status']='ENGINEERING_SMOKE_COMPLETE'
    report['metrics']={k:{'mean_oracle_jsd':float(np.mean(v,axis=0)[0]),'mean_topk':float(np.mean(v,axis=0)[1])} for k,v in m.items()}
    report['explanation_records']=48
    report['model_communication_bytes_per_client']=nledger['model_total_bytes']
    report['static_communication_bytes_per_client']=ledger['static_total_bytes_per_client']
    report['five_round_five_epoch_observed']=True
    (out/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('FINAL_ENGINEERING_SMOKE',json.dumps({'status':report['status'],'accuracy':full['accuracy'],'metrics':report['metrics']}),flush=True)

if __name__=='__main__':main()
