"""Generalized v7 QF-SCAD for real 28x28 and 32x32 images.
77 trainable parameters in the two-convolution spatial correction; preserves
original QF-SCAD width=4 architecture and signed-int8 model snapshots.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from qfscad_impl.core import serialize_state,deserialize_state,jsd_loss

class SpatialCorrection(nn.Module):
    def __init__(self,width=4,power=1.2):
        super().__init__()
        self.conv1=nn.Conv2d(1,width,3,padding=1)
        self.conv2=nn.Conv2d(width,1,3,padding=1)
        nn.init.zeros_(self.conv2.weight);nn.init.zeros_(self.conv2.bias)
        self.power=power
    def forward(self,x,field):
        # x is 1-channel spatial luminance with shape B,1,H,W; never stacked raw RGB.
        s=1.6*torch.tanh(self.conv2(F.gelu(self.conv1(x))).flatten(1))
        base=x.flatten(1).clamp_min(0).pow(self.power)*field
        q=base*torch.exp(s)
        return q/(q.sum(1,keepdim=True)+1.e-12)


def prepare(xx,yy,field,hw):
    X=torch.as_tensor(np.asarray(xx,dtype='float32').reshape(-1,1,*hw).copy())
    Y=torch.as_tensor(np.asarray(yy,dtype='float32').reshape(len(X),-1).copy())
    G=torch.as_tensor(np.tile(np.asarray(field,dtype='float32').reshape(1,-1),(len(X),1)).copy())
    if len(X)<1 or any(not torch.isfinite(z).all() for z in (X,Y,G)) or (Y<0).any():
        raise ValueError('Invalid real-data IG records')
    return X,Y,G


def _fit_local(start_state,triple,steps,lr,seed,wd=0.,device="cpu"):
    torch.manual_seed(seed)
    net=SpatialCorrection()
    net.load_state_dict(deserialize_state(start_state,net.state_dict()))
    net=net.to(device)
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd)
    X,Y,G=(v.to(device) for v in triple)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss=jsd_loss(net(X,G),Y)
        if not torch.isfinite(loss):raise ValueError('Non-finite teacher loss')
        loss.backward();opt.step()
    return serialize_state(net.state_dict())


def train(teacher_x,teacher_y,fields,hw,seed,rounds=8,steps_per_round=30,lr=.02,device="cpu"):
    if len(teacher_x)<2 or len(teacher_x)!=len(teacher_y) or len(teacher_x)!=len(fields):
        raise ValueError('Mismatched teacher clients')
    torch.manual_seed(seed)
    global_net=SpatialCorrection().to(device)
    current=serialize_state(global_net.state_dict())
    triples=[prepare(x,y,g,hw) for x,y,g in zip(teacher_x,teacher_y,fields)]
    n=len(triples);pkt=len(current)
    up=[0]*n;down=[pkt]*n;history=[]
    for r in range(rounds):
        updates=[]
        for i,trip in enumerate(triples):
            serialized=_fit_local(current,trip,steps_per_round,lr,seed+997*r+31*i,device=device)
            if len(serialized)!=pkt:raise ValueError('Incorrect quantized payload')
            up[i]+=len(serialized)
            updates.append(deserialize_state(serialized,global_net.state_dict()))
        aggregate={k:torch.stack([z[k] for z in updates]).mean(0) for k in updates[0]}
        current=serialize_state(aggregate)
        down=[x+len(current) for x in down]
        global_net.load_state_dict(deserialize_state(current,global_net.state_dict()))
        with torch.no_grad():
            history.append(float(np.mean([jsd_loss(global_net(X.to(device),G.to(device)),Y.to(device)).item() for X,Y,G in triples])))
    return global_net,{'up_model_bytes':up,'down_model_bytes':down,
        'model_total_bytes':[x+y for x,y in zip(up,down)],
        'packet_bytes':pkt,'parameter_count':sum(x.numel() for x in global_net.parameters()),
        'rounds':rounds,'local_steps':steps_per_round,'teacher_maps_transmitted':False},history


def train_private(teacher_x,teacher_y,fields,hw,seed,steps=240,lr=.02,device="cpu"):
    """Strong LOCAL control: full precision, same architecture, 48 real teachers,
    same total per-client optimizer steps as federated training. No quantization
    penalty artificially applied to the baseline; no communication.
    """
    out=[]
    for i,(x,y,f) in enumerate(zip(teacher_x,teacher_y,fields)):
        torch.manual_seed(seed)
        model=SpatialCorrection().to(device)
        X,Y,G=(t.to(device) for t in prepare(x,y,f,hw))
        opt=torch.optim.Adam(model.parameters(),lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss=jsd_loss(model(X,G),Y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite private corrector loss')
            loss.backward();opt.step()
        model.eval();out.append(model)
    return out


def predict(model,x,field,hw):
    X=torch.as_tensor(np.asarray(x,dtype='float32').reshape(-1,1,*hw).copy())
    G=torch.as_tensor(np.tile(np.asarray(field,dtype='float32').reshape(1,-1),(len(X),1)).copy())
    model.eval()
    dev=next(model.parameters()).device
    with torch.no_grad():a=model(X.to(dev),G.to(dev)).detach().cpu().numpy()
    if not np.isfinite(a).all():raise ValueError('Nonfinite explanation')
    return a
