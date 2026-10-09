"""QF-SCAD: quantized federated spatial-context attribution distillation.

Only sufficient-statistic bytes (provided by the existing APGF v6 codec) and
quantized miniature model snapshots cross the federated boundary. The server
never receives client images or Integrated Gradients teacher maps in this design.
This DOES NOT furnish a differential-privacy or secure-aggregation guarantee.
"""
from __future__ import annotations
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

POWER=1.2

class SpatialCorrection(nn.Module):
    """77-parameter input-aware correction on top of fixed private/peer prior."""
    def __init__(self, width=4):
        super().__init__()
        self.conv1=nn.Conv2d(1,width,3,padding=1)
        self.conv2=nn.Conv2d(width,1,3,padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self,x,field):
        # By starting with a zero final layer, training initially exactly
        # reproduces the quantized fixed-sharing prior up to numeric precision.
        s=1.6*torch.tanh(self.conv2(functional.gelu(self.conv1(x))).flatten(1))
        base=(x.flatten(1).clamp_min(0)**POWER)*field.reshape(-1,field.shape[-1])
        q=base*torch.exp(s)
        return q/(q.sum(1,keepdim=True)+1e-12)

def jsd_loss(p,y):
    z=(p+y)/2
    return (.5*(p*torch.log((p+1e-12)/(z+1e-12))+y*torch.log((y+1e-12)/(z+1e-12))).sum(-1)).mean()

def serialize_state(state):
    """Per-tensor signed int8 codes and exact 4-byte float scale."""
    buffers=[]
    for name,t in state.items():
        xx=t.detach().cpu().numpy().astype(np.float32).ravel()
        scale=np.float32(max(float(np.max(np.abs(xx))),1e-8)/127)
        code=np.rint(xx/scale).clip(-127,127).astype(np.int8)
        buffers.append(struct.pack('<f',float(scale))+code.tobytes())
    return b''.join(buffers)

def deserialize_state(packet, template):
    out={};offset=0
    for name,t in template.items():
        n=t.numel()
        if offset+4+n>len(packet):raise ValueError('short serialized state')
        scale=struct.unpack_from('<f',packet,offset)[0];offset+=4
        if not np.isfinite(scale) or scale<=0:raise ValueError('bad quantization scale')
        code=np.frombuffer(packet,dtype=np.int8,count=n,offset=offset).astype(np.float32).copy();offset+=n
        out[name]=torch.from_numpy((code*scale).reshape(t.shape).copy())
    if offset!=len(packet):raise ValueError('unexpected bytes in serialized state')
    return out

def _prepare(x,y,g):
    n=len(x)
    a=torch.as_tensor(np.asarray(x,dtype=np.float32).reshape(n,1,28,28).copy())
    b=torch.as_tensor(np.asarray(y,dtype=np.float32).reshape(n,784).copy())
    z=torch.as_tensor(np.repeat(np.asarray(g,dtype=np.float32).reshape(1,784),n,axis=0).copy())
    if n<1 or not np.isfinite(a.numpy()).all() or not np.isfinite(b.numpy()).all() or not np.isfinite(z.numpy()).all():
        raise ValueError('invalid genuine-MNIST IG teacher record')
    return a,b,z

def federated_fit(teachers_x,teachers_y,fields,rounds=8,local_steps=30,seed=123,lr=.02,width=4,consistency_penalty=0.,include_field_wire=True):
    """FedAvg over quantized model snapshots; all local teacher data stay local.

    fields are the *decoded* APGF-v6 quantized prior gain maps, one per client.
    Actual APGF field statistics exchange is accounted separately, once.
    """
    nclients=len(teachers_x)
    if nclients<2 or len(teachers_y)!=nclients or len(fields)!=nclients:raise ValueError('need matching clients')
    if rounds<1 or local_steps<1:raise ValueError('invalid training budget')
    torch.set_num_threads(4)
    triples=[_prepare(x,y,g) for x,y,g in zip(teachers_x,teachers_y,fields)]
    torch.manual_seed(seed)
    server=SpatialCorrection(width=width)
    current=serialize_state(server.state_dict())
    # Two APGF full-resolution stats packets (790 each), then one gain field (786).
    # Packet sizes supplied by existing codec; assertions belong in the caller.
    initial_up=1580 if include_field_wire else 0
    initial_down=786 if include_field_wire else 0
    per_up=[initial_up]*nclients
    per_down=[initial_down+len(current)]*nclients  # initial model broadcast
    history=[]
    for rnd in range(rounds):
        updates=[]
        for i,(xx,yy,gain) in enumerate(triples):
            client=SpatialCorrection(width=width)
            client.load_state_dict(deserialize_state(current,client.state_dict()))
            optimizer=torch.optim.Adam(client.parameters(),lr=lr)
            for _ in range(local_steps):
                optimizer.zero_grad()
                pred=client(xx,gain)
                loss=jsd_loss(pred,yy)
                if consistency_penalty:
                    prior=(xx.flatten(1).clamp_min(0)**POWER)*gain
                    prior=prior/(prior.sum(1,keepdim=True)+1e-12)
                    loss=loss+consistency_penalty*jsd_loss(pred,prior)
                loss.backward();optimizer.step()
            packet=serialize_state(client.state_dict())
            per_up[i]+=len(packet)
            updates.append(deserialize_state(packet,client.state_dict()))
        aggregated={k:torch.stack([u[k] for u in updates]).mean(0) for k in updates[0]}
        current=serialize_state(aggregated)
        per_down=[a+len(current) for a in per_down]
        server.load_state_dict(deserialize_state(current,server.state_dict()))
        with torch.no_grad():
            history.append(float(np.mean([jsd_loss(server(x,g),y).item() for x,y,g in triples])))
    ledger={'per_client_uplink_bytes':per_up,'per_client_downlink_bytes':per_down,
            'per_client_total_bytes':[a+b for a,b in zip(per_up,per_down)],
            'per_tensor_int8_with_float32_scale':True,
            'privacy_guarantee':False,'n_params':sum(p.numel() for p in server.parameters()),
            'one_model_packet_bytes':len(current),'training_rounds':rounds,'local_steps_per_client_round':local_steps}
    return server, ledger, history

def private_fit(x,y,field,steps=240,seed=123,width=4,lr=.02):
    torch.set_num_threads(4)
    xx,yy,g=_prepare(x,y,field)
    torch.manual_seed(seed)
    model=SpatialCorrection(width=width)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    for _ in range(steps):
        opt.zero_grad();ll=jsd_loss(model(xx,g),yy);ll.backward();opt.step()
    return model

def predict(model,x,gain):
    n=len(x)
    xx=torch.tensor(np.asarray(x,dtype=np.float32).reshape(n,1,28,28).copy())
    gg=torch.tensor(np.repeat(np.asarray(gain,dtype=np.float32).reshape(1,784),n,axis=0).copy())
    with torch.no_grad():return model(xx,gg).detach().cpu().numpy()
