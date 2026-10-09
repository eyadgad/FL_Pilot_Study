"""Dimension-independent byte-serialized positive gradient-field codec.
Only bytes cross the simulated client/server boundary. No DP guarantee.
"""
from __future__ import annotations
import struct
import numpy as np
from scipy.ndimage import gaussian_filter


def norm(x):
    a=np.maximum(np.asarray(x,dtype=np.float64),0)
    return a/(a.sum(axis=-1, keepdims=True)+1.e-12)


def _check(a, d):
    a=np.asarray(a,dtype=np.float64).reshape(-1)
    if a.size!=d or not np.isfinite(a).all() or (a<0).any():
        raise ValueError('Invalid nonnegative attribution / occupancy statistics')
    return a


def stat_encode(a,n):
    a=_check(a,len(np.asarray(a).reshape(-1)))
    if n<1 or n>65535:raise ValueError('Invalid count')
    scale=np.float32(max(float(a.max())/255,1e-30))
    raw=np.rint(a/scale).clip(0,255).astype('u1')
    return struct.pack('<Hf',n,float(scale))+raw.tobytes()


def stat_decode(pkt,d):
    if len(pkt)!=6+d:raise ValueError('Wrong stat packet length')
    n,s=struct.unpack('<Hf',pkt[:6]);
    if n<1 or not np.isfinite(s) or s<=0:raise ValueError('Corrupt stat header')
    return np.frombuffer(pkt[6:],dtype=np.uint8).astype('float64')*s,n


def field_encode(field):
    g=np.maximum(_check(field,len(np.asarray(field).reshape(-1))),1e-9)
    g=g/(g.mean()+1e-9)
    if g.size>65535:raise ValueError('Too many pixels')
    code=np.rint((np.clip(np.log(g),-3,3)+3)/6*255).astype(np.uint8)
    return struct.pack('<H',g.size)+code.tobytes()


def field_decode(pkt,d):
    if len(pkt)!=d+2 or struct.unpack('<H',pkt[:2])[0]!=d:raise ValueError('Wrong field dimensions')
    g=np.exp(-3+np.frombuffer(pkt[2:],dtype=np.uint8).astype('float64')/255*6)
    return g/(g.mean()+1e-9)


def fit_fields(teacher_x,teacher_y,hw,peer_weight=.2,power=1.2,blur=1.):
    """All clients fit exactly the same number of real IG teachers.
    Teacher data -> quantized A/B sums -> server -> quantized per-client fields.
    Peer mixture excludes the requesting client to prevent double weighting.
    """
    d=int(np.prod(hw));coded=[];decoded=[]
    if not teacher_x or len(teacher_x)!=len(teacher_y):raise ValueError('Missing client teachers')
    for xx,yy in zip(teacher_x,teacher_y):
        X=np.asarray(xx).reshape(len(xx),d);Y=norm(np.asarray(yy).reshape(len(yy),d))
        if len(X)<1 or not np.isfinite(X).all() or not np.isfinite(Y).all():raise ValueError('Invalid real teachers')
        n=len(X);a=Y.sum(axis=0);b=np.maximum(X,0).__pow__(power).sum(axis=0)
        p=(stat_encode(a,n),stat_encode(b,n));coded.append(p)
        a1,n1=stat_decode(p[0],d);b1,n2=stat_decode(p[1],d)
        if n1!=n2:raise ValueError('Wrong teacher support')
        decoded.append((a1,b1,n1))
    field_private=[];field_fed=[];field_server=[]
    for i,(a,b,n) in enumerate(decoded):
        ap=sum(z[0] for j,z in enumerate(decoded) if j!=i)
        bp=sum(z[1] for j,z in enumerate(decoded) if j!=i)
        np_=sum(z[2] for j,z in enumerate(decoded) if j!=i)
        for dest,am,bm,nm in [(field_private,a,b,n),
                              (field_fed,a+peer_weight*ap,b+peer_weight*bp,n+peer_weight*np_),
                              (field_server,a+ap,b+bp,n+np_)]:
            g=am/(bm+0.01*nm)
            if blur:g=gaussian_filter(g.reshape(hw),blur).reshape(-1)
            dest.append(field_decode(field_encode(g),d))
    uplink=[sum(len(x) for x in pp) for pp in coded]
    one_down=d+2
    return {'private':np.stack(field_private), 'fixed':np.stack(field_fed),
            'global':np.stack(field_server)}, {'private_bytes_per_client':[0]*len(coded),
          'static_uplink_bytes_per_client':uplink,'static_downlink_bytes_per_client':[one_down]*len(coded),
          'static_total_bytes_per_client':[x+one_down for x in uplink],
          'field_bytes':one_down,'server_received_only_quantized_packets':True,
          'no_formal_privacy_guarantee':True}
