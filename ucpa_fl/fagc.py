"""Federated Attribution Gradient Calibration (FAGC) - experimental.

Real-data only: fit class and global occupancy-conditioned attribution gain maps
from *disjoint* task-model integrated-gradient teacher records. 14x14 log-gain
quantized packets. Only byte packets cross simulated federation boundary.
No formal privacy guarantee, no theorem of explanation validity.
"""
from __future__ import annotations
import struct
import numpy as np
from scipy.ndimage import gaussian_filter
from .metrics import normalize_np

SIDE=14; GRID=196; LO=-3.; HI=3.

def _norm_gain(g):
    g=np.maximum(np.asarray(g,float),1e-9)
    return g/(np.mean(g)+1e-9)

def gain_from_sufficient(A,B,n,sigma=1.5):
    if not n:return np.ones(784)
    field=(A/(B+0.01*n)).reshape(28,28)
    if sigma:field=gaussian_filter(field,sigma)
    field=field.reshape(14,2,14,2).mean((1,3))
    return _norm_gain(field).ravel()

def encode_gain(field,n):
    """2-byte support followed by 196 uint8 log-field values; empty sends support only."""
    if n<0 or n>65535:raise ValueError('support out of range')
    if not n:return struct.pack('<H',0)
    v=np.maximum(_norm_gain(field),1e-8)
    q=np.rint((np.clip(np.log(v),LO,HI)-LO)/(HI-LO)*255).astype(np.uint8)
    return struct.pack('<H',int(n))+q.tobytes()

def decode_gain(packet):
    n=struct.unpack('<H',packet[:2])[0]
    if not n:
        if len(packet)!=2:raise ValueError('zero-support packet')
        return np.ones(GRID),0
    if len(packet)!=2+GRID:raise ValueError('packet size mismatch')
    q=np.frombuffer(packet[2:],dtype=np.uint8).astype(float)
    field=np.exp(LO+q/255*(HI-LO))
    return _norm_gain(field),n

def fit_field_packets(teacher_records,n_classes=10,power=1.2,sigma=1.5):
    N=len(teacher_records); packets=[]
    for i,recs in enumerate(teacher_records):
        by=[[] for _ in range(n_classes)]
        for rec in recs:
            c=int(rec['pred']);
            if c<0 or c>=n_classes:raise ValueError('Invalid predicted class')
            by[c].append(rec)
        cp=[]
        for row in [list(recs),*by]:
            A=np.zeros(784);B=np.zeros(784)
            for rec in row:
                xx=np.asarray(rec['x']).reshape(-1).astype(float)
                oo=normalize_np(np.asarray(rec['oracle_map'])[None])[0]
                A+=oo;B+=np.maximum(xx,0)**power
            cp.append(encode_gain(gain_from_sufficient(A,B,len(row),sigma),len(row)))
        packets.append(cp)
    return packets

def aggregate_packets(packets,global_mix=.5,self_strength=.2):
    """Server uses decoded class summaries, not original teacher records.

    Weighted log-barycenter; global fallback for absent/rare classes.
    Output: per-client,class 14x14 *log field* codec packet and byte ledger.
    """
    N=len(packets);C=len(packets[0])-1
    uplink=np.asarray([sum(map(len,row)) for row in packets]);
    decoded=[[decode_gain(b) for b in row] for row in packets]
    global_fields=np.stack([decoded[i][0][0] for i in range(N)])
    global_n=np.array([decoded[i][0][1] for i in range(N)])
    glog=np.average(np.log(global_fields),axis=0,weights=np.maximum(global_n,1))
    downloads=[];full=np.zeros((N,C,784));dn=[]
    for i in range(N):
        row=[]
        for c in range(C):
            cl=[];wc=[]
            for k in range(N):
                g,n=decoded[k][c+1]
                if n>0:
                    cl.append(np.log(g));wc.append(n)
            cfield=np.average(np.stack(cl),axis=0,weights=wc) if cl else glog
            own,ownN=decoded[i][c+1]
            ownlog=np.log(own) if ownN else cfield
            logfield=(1-global_mix)*cfield+global_mix*glog
            if ownN:logfield=(1-self_strength)*logfield+self_strength*ownlog
            payload=encode_gain(np.exp(logfield),1)
            # use actually decoded downlink, not a float oracle/noise-free assumption
            q,_=decode_gain(payload)
            full[i,c]=np.repeat(np.repeat(q.reshape(14,14),2,axis=0),2,axis=1).reshape(-1)
            row.append(payload)
        downloads.append(row);dn.append(sum(map(len,row)))
    return full,{'uplink_bytes':uplink.tolist(),'downlink_bytes':dn,'packet_type':'uint16 count + 196 uint8 14x14 quantized log gains, no DP',
          'source':'only quantized client packets reach server','n_clients':N,'n_classes':C}

def maps_from_x(X,pred,client,field,power=1.2,strength=.75):
    x=np.asarray(X,dtype=float).reshape(-1,784)
    idx=np.asarray(client,int);yc=np.asarray(pred,int)
    return normalize_np((np.maximum(x,0)**power)*(field[idx,yc]**strength))

# --- Class-agnostic sufficient-statistic channel: 2 x 28x28 byte maps uplink ---
def encode_stat(x,n):
    """Quantize nonnegative sum to 8bits, carrying exact support and float32 scale."""
    A=np.asarray(x,dtype=float).reshape(-1)
    if A.shape!=(784,) or not np.all(np.isfinite(A)) or np.min(A)<0: raise ValueError('bad statistic')
    if n<1 or n>65535:raise ValueError('bad teacher count')
    mx=np.float32(float(A.max())/255.)
    if mx<=0:raise ValueError('zero statistic')
    q=np.rint(A/mx).clip(0,255).astype(np.uint8)
    return struct.pack('<Hf',int(n),float(mx))+q.tobytes()

def decode_stat(packet):
    if len(packet)!=790:raise ValueError('expected 790-byte stat packet')
    count,scale=struct.unpack('<Hf',packet[:6])
    if count<1 or not np.isfinite(scale) or scale<=0:raise ValueError('invalid support/scale')
    return np.frombuffer(packet[6:],dtype=np.uint8).astype(float)*scale,count

def train_fagc(teachers,power=1.2,sigma=1.0,peer_fraction=.2):
    """Compress genuine-MNIST IG teacher sufficient statistics before aggregation.

    Never gives server IG maps, client images, local surrogate weights, or labels.
    The only server-visible data are quantized A/B statistics.
    Outputs quantized 28x28 gain field decoded after client downlink.
    """
    N=len(teachers);packets=[]
    for i,recs in enumerate(teachers):
        if not recs:raise ValueError('no teacher records')
        A=np.zeros(784);B=np.zeros(784)
        for r in recs:
            xx=np.asarray(r['x']).reshape(-1).astype(float)
            oo=normalize_np(np.asarray(r['oracle_map'])[None])[0]
            if xx.shape!=(784,) or oo.shape!=(784,):raise ValueError('bad real-MNIST image/IG shape')
            A+=oo;B+=np.maximum(xx,0)**power
        packets.append((encode_stat(A,len(recs)),encode_stat(B,len(recs))))
    # Server must aggregate only the received bytes.
    decoded=[]
    for a,b in packets:
        AA,na=decode_stat(a);BB,nb=decode_stat(b)
        if na!=nb:raise ValueError('support mismatch')
        decoded.append((AA,BB,na))
    allA=sum(row[0] for row in decoded);allB=sum(row[1] for row in decoded)
    allN=sum(row[2] for row in decoded)
    fields=[];down=[]
    for A,B,n in decoded:
        gain=(A+peer_fraction*allA)/(B+peer_fraction*allB+0.01*(n+peer_fraction*allN))
        if sigma:gain=gaussian_filter(gain.reshape(28,28),sigma).ravel()
        # log gain packet: exact 784-byte uint8 values and 2-byte count
        payload=encode_gain_full(gain)
        field=decode_gain_full(payload)
        fields.append(field);down.append(len(payload))
    codec={'packet_type':'uplink: each client uint16 teacher count + float32 scale + 784 uint8 per 2 sufficient statistics; downlink uint16 + 784 quantized log-field uint8',
           'uplink_bytes':[sum(map(len,p)) for p in packets], 'downlink_bytes':down,
           'teacher_counts':[row[2] for row in decoded],
           'teacher_budget_equal_to_private':True,'image_source':'real MNIST CSV; no synthetic samples',
           'privacy_claim':False}
    return np.asarray(fields),codec

def encode_gain_full(f):
    g=_norm_gain(np.asarray(f).ravel())
    if g.shape!=(784,):raise ValueError('field needs 784 coordinates')
    q=np.rint((np.clip(np.log(np.maximum(g,1e-9)),LO,HI)-LO)/(HI-LO)*255).astype(np.uint8)
    return struct.pack('<H',784)+q.tobytes()

def decode_gain_full(b):
    if len(b)!=786 or struct.unpack('<H',b[:2])[0]!=784:raise ValueError('invalid gain wire packet')
    v=np.exp(LO+np.frombuffer(b[2:],dtype=np.uint8).astype(float)/255*(HI-LO))
    return _norm_gain(v)

def fagc_map(x,local_map,gain,power=1.2,class_exponent=.05):
    xx=np.maximum(np.asarray(x,float).reshape(-1),0)
    ll=normalize_np(np.asarray(local_map,float).reshape(1,-1))[0]
    if xx.shape!=(784,) or ll.shape!=(784,):raise ValueError('only 28x28 real MNIST')
    class_factor=((ll+1e-8)/(xx+0.03))**class_exponent
    return normalize_np((xx**power*np.asarray(gain,float)*class_factor)[None])[0]
