import json, numpy as np, torch
from pathlib import Path
from ucpa_fl.alignment import jsd, ucpa_prior
from ucpa_fl.artifacts import sanitize_artifact
from ucpa_fl.config import AlignmentConfig, ArtifactConfig, ExperimentConfig
from ucpa_fl.datasets import _split_client_indices, iid_partition
from ucpa_fl.logging_utils import RunLogger
from ucpa_fl.repro import environment_snapshot, move_state_dict, resolve_device


def test_auto_device_uses_cuda_only_when_visible():
    dev=resolve_device('auto')
    assert dev.type==('cuda' if torch.cuda.is_available() else 'cpu')
    assert resolve_device('gpu').type==dev.type
    moved=move_state_dict({'w':torch.zeros(2)}, dev)
    assert moved['w'].device.type==dev.type
    if not torch.cuda.is_available():
        import pytest
        with pytest.raises(RuntimeError):
            resolve_device('cuda')


def test_client_split_disjoint_and_complete():
    idx=np.arange(100); parts=_split_client_indices(idx,7)
    merged=np.concatenate(parts)
    assert len(merged)==100 and len(np.unique(merged))==100
    assert set(merged)==set(idx)


def test_artifact_topk_and_normalization():
    rng=np.random.default_rng(1); mean=rng.random((3,20)); mean/=mean.sum(1,keepdims=True)
    var=np.ones_like(mean)*1e-3; counts=np.array([20,20,20])
    cfg=ArtifactConfig(topk=5,dp_sigma=0.0,quant_bits=8)
    a=sanitize_artifact(mean,var,counts,cfg,rng)
    assert np.all(a.mask.sum(1)==5)
    assert np.allclose(a.mean.sum(1),1.0)
    assert a.bytes_estimate>0


def test_ucpa_homogeneous_peers_pool():
    means=np.array([[[.6,.4]],[[.59,.41]],[[.61,.39]]],float)
    var=np.ones_like(means)*.01; masks=np.ones_like(means,bool)
    cfg=AlignmentConfig(ucpa_z=2,ucpa_h=.048,ucpa_variance_floor_fraction=.05)
    cfg.sparse_intersection_only=True
    p,e=ucpa_prior(means,var,masks,cfg)
    assert np.all(e>1.2)
    assert np.max(np.abs(p[:,0,0]-.6)) < .02


def test_ucpa_rejects_incompatible_coordinate():
    means=np.array([[[.9,.1]],[[.1,.9]]],float)
    var=np.ones_like(means)*1e-6; masks=np.ones_like(means,bool)
    cfg=AlignmentConfig(ucpa_z=2,ucpa_h=.048,ucpa_variance_floor_fraction=.05); cfg.sparse_intersection_only=True
    p,e=ucpa_prior(means,var,masks,cfg)
    assert p[0,0,0] > .85 and p[1,0,1] > .85


def test_logger_integrity_is_not_stale(tmp_path):
    log=RunLogger(tmp_path,'t',1,{'x':1},{'python':'test'})
    log.metric('m',1.0); log.finalize()
    run=log.run_dir
    for line in (run/'integrity.sha256').read_text().splitlines():
        digest,rel=line.split('  ',1)
        import hashlib
        assert hashlib.sha256((run/rel).read_bytes()).hexdigest()==digest


def test_ucpa_matches_smoke_reference_dense():
    rng=np.random.default_rng(55)
    E=rng.random((4,1,11));E/=E.sum(2,keepdims=True)
    V=rng.random((4,1,11))*2e-3
    masks=np.ones_like(E,dtype=bool)
    cfg=AlignmentConfig(ucpa_z=2.0,ucpa_h=.048,ucpa_variance_floor_fraction=.05);cfg.sparse_intersection_only=True
    got,_=ucpa_prior(E,V,masks,cfg)
    # Independent transcription of the smoke-approved phase-5 operator.
    X=E[:,0].copy(); X/=X.sum(1,keepdims=True); VV=V[:,0]
    pos=VV[VV>0];floor=float(np.median(pos))*.05 if len(pos) else 1e-12
    diff=X[:,None,:]-X[None,:,:];denom=np.sqrt(VV[:,None,:]+VV[None,:,:]+floor+1e-15)
    coord=np.exp(-.5*((np.abs(diff)/denom)/2.0)**2)
    J=np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for k in range(i+1,len(X)):
            J[i,k]=J[k,i]=jsd(X[i],X[k])
    G=np.exp(-J/.048);W=coord*G[:,:,None]
    for i in range(len(X)):W[i,i,:]=1
    ref=np.einsum('ikd,kd->id',W,X)/np.maximum(W.sum(1),1e-15);ref/=ref.sum(1,keepdims=True)
    assert np.allclose(got[:,0],ref,atol=1e-11,rtol=1e-10)
