"""Experimental hierarchical residual correction for federated explanations.

Private teacher budget is identical across private/shared/LOO/EB-LOO policies.
Teacher is direct task-model IG, a reference rather than causal ground truth.
Shared corrections are compressed before use. This is NOT a privacy guarantee.
"""
from __future__ import annotations
import numpy as np
from .metrics import normalize_np


def sparse_signed(v, k=64):
    arr=np.asarray(v, dtype=np.float64)
    out=np.zeros_like(arr)
    k=min(max(0,int(k)),arr.shape[-1])
    if k:
        ind=np.argpartition(np.abs(arr),-k,axis=-1)[..., -k:]
        np.put_along_axis(out,ind,np.take_along_axis(arr,ind,axis=-1),axis=-1)
    return out


def fit_residual_policies(records_per_client, n_classes:int, dim:int, topk:int=64):
    """Fits four policies from identical private calibration records.

    LOO excludes own client from peer correction at every class; no eval records.
    EB weights use nonnegative method-of-moments cross-client variance, with
    a within-client sampling variance estimate. No formal posterior claim.
    """
    N=len(records_per_client);C=int(n_classes);D=int(dim)
    counts=np.zeros((N,C),np.int64);means=np.zeros((N,C,D));noise=np.ones((N,C))
    for i,recs in enumerate(records_per_client):
        by=[[] for _ in range(C)]
        for r in recs:
            c=int(r['pred'])
            if 0<=c<C:
                l=normalize_np(np.asarray(r['local_map'],float)[None])[0]
                o=normalize_np(np.asarray(r['oracle_map'],float)[None])[0]
                by[c].append(o-l)
        for c,vals in enumerate(by):
            if not vals:continue
            X=np.asarray(vals,float);counts[i,c]=len(X);means[i,c]=X.mean(axis=0)
            noise[i,c]=float(np.mean(np.var(X,axis=0,ddof=1)) / len(X)) if len(X)>1 else 0.1
    shared=np.zeros((C,D));loo=np.zeros((N,C,D));eb=np.zeros((N,C,D));fixed=np.zeros((N,C,D));weights=np.zeros((N,C))
    for c in range(C):
        n=counts[:,c]
        if n.sum(): shared[c]=(means[:,c]*n[:,None]).sum(axis=0)/n.sum()
        for i in range(N):
            peers=n.copy();peers[i]=0
            if peers.sum():loo[i,c]=(means[:,c]*peers[:,None]).sum(axis=0)/peers.sum()
            else: loo[i,c]=0
            valid=(n>0)&(np.arange(N)!=i)
            if valid.sum()>=2:
                across=np.var(means[valid,c],axis=0,ddof=1).mean()
                tau2=max(0.,float(across-np.mean(noise[valid,c])))
            else:tau2=0.
            ni=int(n[i]);ni_noise=float(noise[i,c]) if ni else float('inf')
            w=(tau2/(tau2+ni_noise+1e-12)) if np.isfinite(ni_noise) else 0.
            weights[i,c]=float(np.clip(w,0,1))
            eb[i,c]=w*means[i,c]+(1-w)*loo[i,c]
            wf=ni/(ni+2.)
            fixed[i,c]=wf*means[i,c]+(1-wf)*shared[c]
    policies={'private':sparse_signed(means,topk),'shared':np.broadcast_to(sparse_signed(shared,topk),(N,C,D)).copy(),
      'loo':sparse_signed(loo,topk),'ebloo':sparse_signed(eb,topk),'fixed':sparse_signed(fixed,topk)}
    return policies,{'counts':counts.tolist(),'weights':weights.tolist(),
      'n_teacher_per_client':[len(r) for r in records_per_client],
      'topk':int(topk),'number_of_clients':N,'n_classes':C,
      'signed_sparse_wire_bytes_per_class':int(min(topk,D)*((2 if D<=65535 else 4)+1)+4+4),
      'note':'8-bit signed value plus per-class float32 scale, count, and 16/32-bit indices; byte estimate; no codec/privacy claim'}


def corrected_map(local,prior,residual,alpha=0.6,beta=0.4):
    l=normalize_np(np.asarray(local,float)[None])[0]
    p=normalize_np(np.asarray(prior,float)[None])[0]
    v=normalize_np(np.maximum(l+float(alpha)*np.asarray(residual,float),0)[None])[0]
    return normalize_np(((1-float(beta))*v+float(beta)*p)[None])[0]


def wire_encode_class(residual, count:int, topk=64, noise:float=0.0):
    """Real sparse-index/signed-int8 protocol: (nuint16, nobs uint32, scale f32) + entries.
    Not cryptographically private and not differentially private.
    """
    import struct
    x=np.asarray(residual,float).reshape(-1);D=x.size
    k=min(int(topk),D) if int(count)>0 else 0
    if k:
        idx=np.argpartition(np.abs(x),-k)[-k:]
        idx=np.sort(idx).astype('<u2' if D<=65535 else '<u4')
        chosen=x[idx];scale=max(float(np.max(np.abs(chosen))),1e-12)
        vals=np.rint(np.clip(chosen/scale,-1,1)*127).astype('i1')
    else:
        idx=np.array([],dtype='<u2' if D<=65535 else '<u4');vals=np.array([],dtype='i1');scale=1.
    return struct.pack('<HIff',k,int(count),scale,float(noise))+idx.tobytes()+vals.tobytes()


def wire_decode_class(packet:bytes,dim:int):
    import struct
    if len(packet)<14:raise ValueError('short packet')
    k,n,s,sample_noise=struct.unpack('<HIff',packet[:14]); idx_size=2 if dim<=65535 else 4
    if len(packet)!=14+k*(idx_size+1):raise ValueError('packet length mismatch')
    idx=np.frombuffer(packet[14:14+k*idx_size],dtype='<u2' if idx_size==2 else '<u4')
    val=np.frombuffer(packet[14+k*idx_size:],dtype='i1')
    if np.any(idx>=dim) or len(np.unique(idx))!=len(idx):raise ValueError('bad index')
    out=np.zeros(dim,float);out[idx]=val.astype(float)*float(s)/127
    return out,int(n),float(sample_noise)


def fit_wire_residual_policies(records_per_client, n_classes:int, dim:int, topk:int=64):
    """Fit policies using what a server *actually receives*: codec-decoded means.
    Private method uses unquantized local class means; all have same IG budget.
    """
    N=len(records_per_client);C=int(n_classes);D=int(dim)
    direct=np.zeros((N,C,D)); counts=np.zeros((N,C),int); noise=np.ones((N,C),float)
    packets=[];bytes_up=np.zeros(N,int)
    for i,recs in enumerate(records_per_client):
        by=[[] for _ in range(C)]
        for r in recs:
            c=int(r['pred'])
            if 0<=c<C:
                l=normalize_np(np.asarray(r['local_map'],float)[None])[0]
                o=normalize_np(np.asarray(r['oracle_map'],float)[None])[0]
                by[c].append(o-l)
        client_packets=[]
        for c,X in enumerate(by):
            if X:
                X=np.asarray(X);counts[i,c]=len(X);direct[i,c]=X.mean(0)
                noise[i,c]=float(np.var(X,axis=0,ddof=1).mean()/len(X)) if len(X)>1 else .1
            packet=wire_encode_class(direct[i,c],counts[i,c],topk,noise[i,c])
            client_packets.append(packet);bytes_up[i]+=len(packet)
        packets.append(client_packets)
    wire=np.zeros_like(direct);wire_noise=np.ones_like(noise)
    for i in range(N):
        for c in range(C):
            wire[i,c],nc,wire_noise[i,c]=wire_decode_class(packets[i][c],D)
            assert nc==counts[i,c]
    shared=np.zeros((N,C,D));loo=np.zeros_like(shared);eb=np.zeros_like(shared);fixed=np.zeros_like(shared);weights=np.zeros((N,C))
    for c in range(C):
        nc=counts[:,c];total=int(nc.sum())
        mean=((wire[:,c]*nc[:,None]).sum(0)/total) if total else np.zeros(D)
        for i in range(N):
            shared[i,c]=mean
            others=nc.copy();others[i]=0
            peer=((wire[:,c]*others[:,None]).sum(0)/others.sum()) if others.sum() else np.zeros(D)
            loo[i,c]=peer
            valid=(nc>0)&(np.arange(N)!=i)
            tau2=max(0.,float(np.var(wire[valid,c],axis=0,ddof=1).mean()-np.mean(noise[valid,c]))) if valid.sum()>1 else 0.
            v=float(wire_noise[i,c]) if nc[i] else np.inf
            w=float(tau2/(tau2+v+1e-12)) if np.isfinite(v) else 0.
            weights[i,c]=w
            eb[i,c]=w*wire[i,c]+(1-w)*peer
            f=float(nc[i]/(nc[i]+2.))
            fixed[i,c]=f*wire[i,c]+(1-f)*mean
    # Keep client-private correction unquantized; it never goes on wire.
    pol={'private':sparse_signed(direct,topk), 'shared':sparse_signed(shared,topk),
      'loo':sparse_signed(loo,topk), 'ebloo':sparse_signed(eb,topk), 'fixed':sparse_signed(fixed,topk)}
    return pol,{'counts':counts.tolist(),'shrinkage_weights':weights.tolist(),
      'n_teacher_per_client':[len(r) for r in records_per_client], 'bytes_per_client_up':bytes_up.tolist(),
      'bytes_per_client_down':bytes_up.tolist(), 'protocol':'class-ordered signed topk int8, uint16/uint32 index, class count uint32, scale float32, noise float32; no DP',
      'client_calibration_disjoint_from_eval':True,'topk':int(topk)}
