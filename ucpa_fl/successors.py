"""Exploratory, fixed-budget federated explanation-residual successor candidates.

All inputs are real-data IG/surrogate calibration records, disjoint from evaluation.
Codec uses the v3 signed int8 sparse class packet. No DP/security guarantee.
Unlike a privacy-preserving protocol, these messages may leak attribution data.
"""
from __future__ import annotations
import numpy as np
import torch
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from scipy.fft import dctn, idctn
from .metrics import normalize_np
from .residual import wire_encode_class,wire_decode_class,sparse_signed


def _angle(kind, client, nclients, max_deg):
    if kind != 'rotation': return 0.
    f=0 if client==0 else client/max(1,nclients-1)
    return float(-max_deg+2*max_deg*f)


def _rotate(x, angle):
    # Bilinear, not exact inverse: this is a hypothesis to test, not an assertion.
    if abs(angle)<1e-9:return x.copy()
    a=torch.as_tensor(x.reshape(1,28,28),dtype=torch.float32)
    return rotate(a,angle,interpolation=InterpolationMode.BILINEAR).numpy().reshape(-1).astype(np.float64)


def _haar_same_budget(residual, k):
    # A deterministic transform-domain ablation; DCT yields a spatial residual after inverse.
    y=dctn(np.asarray(residual).reshape(28,28),type=2,norm='ortho').ravel()
    z=sparse_signed(y,k).reshape(28,28)
    return idctn(z,type=2,norm='ortho').ravel()


def make_successors(teacher_records, n_classes, dim, topk, shift_kind, max_deg):
    """Return {name: [client,class,pixel]} and exact serialized class-packet accounting.

    All methods have 12-or-fewer private IG teacher records per client; own classes with
    zero observations remain unsupported and are never fabricated. Server uses only
    codec-decoded sparse int8 residuals. Private method uses local full-precision means.
    """
    if dim != 784: raise ValueError('successors currently require real 28x28 MNIST')
    N=len(teacher_records); C=int(n_classes); D=int(dim); k=int(topk)
    counts=np.zeros((N,C),np.int64); direct=np.zeros((N,C,D));wire=np.zeros_like(direct)
    upload=np.zeros(N,int); wire_packets=[]
    for i,recs in enumerate(teacher_records):
        vals=[[] for _ in range(C)]
        for r in recs:
            c=int(r['pred'])
            if 0<=c<C:
                o=normalize_np(np.asarray(r['oracle_map'])[None])[0]
                l=normalize_np(np.asarray(r['local_map'])[None])[0]
                vals[c].append(o-l)
        row=[]
        for c,X in enumerate(vals):
            counts[i,c]=len(X)
            if X:direct[i,c]=np.mean(X,axis=0)
            p=wire_encode_class(direct[i,c],counts[i,c],k)
            v,n,_=wire_decode_class(p,D)
            if n !=counts[i,c]:raise AssertionError('codec count mismatch')
            wire[i,c]=v; row.append(p); upload[i]+=len(p)
        wire_packets.append(row)
    private=sparse_signed(direct,k)
    fixed=np.zeros_like(direct); sarr=np.zeros_like(direct); gera=np.zeros_like(direct); lrsi=np.zeros_like(direct); dct=np.zeros_like(direct)
    angles=np.array([_angle(shift_kind,i,N,max_deg) for i in range(N)])
    canonical=np.stack([np.stack([_rotate(wire[i,c],-angles[i]) for c in range(C)]) for i in range(N)])
    for c in range(C):
        n=counts[:,c]; total=n.sum()
        mean=(wire[:,c]*n[:,None]).sum(0)/total if total else np.zeros(D)
        can=(canonical[:,c]*n[:,None]).sum(0)/total if total else np.zeros(D)
        # No basis communication: rank-one factor is computed from decoded uploads.
        supported=np.where(n>0)[0]
        if len(supported)>1:
            mat=wire[supported,c]
            u,s,vh=np.linalg.svd(mat,full_matrices=False)
            rankone=(u[:,:1]*s[:1]) @ vh[:1,:]
            rank_map={int(i):rankone[j] for j,i in enumerate(supported)}
        else: rank_map={int(i):wire[i,c] for i in supported}
        for i in range(N):
            if not n[i]:continue
            own=wire[i,c]; peers=n.copy();peers[i]=0
            peer=(wire[:,c]*peers[:,None]).sum(0)/peers.sum() if peers.sum() else own
            ni=float(n[i]); lam=ni/(ni+2.)
            fixed[i,c]=sparse_signed(lam*own+(1-lam)*mean,k)
            # SARR: conservative, support- and disagreement-aware cross-client borrowing.
            reliability=(peers.sum()/(peers.sum()+8.))*(ni/(ni+4.))
            disagreement=np.sum(np.abs(own-peer))
            borrow=float(np.clip(reliability*np.exp(-4.*disagreement),0,.5)) if peers.sum() else 0.
            sarr[i,c]=sparse_signed((1-borrow)*own+borrow*peer,k)
            # GERA: transport residual coordinates into canonical image geometry.
            local_can=canonical[i,c]
            geometric=_rotate(can,angles[i])
            # Explicit own support required. Pooling here includes own contribution.
            gera[i,c]=sparse_signed(lam*own+(1-lam)*geometric,k)
            # LRSI: low-rank common component plus own sparse innovation (fixed coeff).
            common=rank_map.get(i,own)
            lrsi[i,c]=sparse_signed(.5*own+.5*common,k)
            # Transform-domain coder: ablation only, with identical per-class packet format.
            dct[i,c]=sparse_signed(_haar_same_budget(own,k),k)
    result={'private':private,'fixed':fixed,'sarr':sarr,'gera':gera,'lrsi':lrsi,'dct':dct}
    result={name:sparse_signed(arr,k) for name,arr in result.items()}
    # Exact received bytes for the same payload type, not an estimated 'number of floats'.
    down={}
    for name,arr in result.items():
        if name=='private': down[name]=[0]*N;continue
        down[name]=[sum(len(wire_encode_class(arr[i,c],int(counts[i,c]),k)) for c in range(C)) for i in range(N)]
    return result, {'teacher_counts':counts.tolist(),'teachers_per_client':[len(x) for x in teacher_records],
        'upload_bytes_per_client':upload.tolist(),'download_bytes_per_client':down,
        'codec':'uint16 topk, uint32 count, float32 scale, float32 sample-noise, uint16 indices, signed int8 values',
        'no_privacy_guarantee':True,'same_private_IG_budget':True, 'topk':k,
        'privacy_note':'Teacher records stay client-side; signed sparse uploads are not DP.'}
