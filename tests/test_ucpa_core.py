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

from ucpa_fl.alignment import nc_sacpa_prior, sacpa_global_count_prior

def test_ncsacpa_self_tuning_survives_distance_rescaling():
    # Same geometry at very different absolute JSD scales should retain active peers.
    e1=np.array([[[.50,.30,.20]],[[.48,.32,.20]],[[.20,.30,.50]],[[.18,.32,.50]]],float)
    e2=np.power(e1,3); e2/=e2.sum(2,keepdims=True)
    m=np.ones_like(e1,bool)
    p1,d1=nc_sacpa_prior(e1,m,power=2,neighborhood_size=3,min_missing_support=2)
    p2,d2=nc_sacpa_prior(e2,m,power=2,neighborhood_size=3,min_missing_support=2)
    assert d1['peer_mass_mean']>0.5 and d2['peer_mass_mean']>0.5
    assert np.allclose(p1.sum(2),1) and np.allclose(p2.sum(2),1)

def test_ncsacpa_local_corroboration_blocks_foreign_unique_support():
    # Client 0 lacks feature 3. Only one of its three closest peers reports it,
    # so local corroboration must keep that coordinate absent.
    E=np.array([
      [[.55,.30,.15,0]],
      [[.54,.31,.15,0]],
      [[.56,.29,.15,0]],
      [[.20,.20,.10,.50]],
    ],float)
    M=E>0
    p,_=nc_sacpa_prior(E,M,power=2,neighborhood_size=3,min_missing_support=2)
    assert p[0,0,3] < 1e-12

def test_ncsacpa_differs_from_global_count_on_grouped_support():
    E=np.array([
      [[.60,.40,0,0]],[[.58,.42,0,0]],[[.62,.38,0,0]],
      [[.25,.25,.25,.25]],[[.24,.26,.25,.25]],[[.26,.24,.25,.25]],
    ],float)
    M=E>0
    a,_=sacpa_global_count_prior(E,M,power=2,min_missing_support=2)
    b,_=nc_sacpa_prior(E,M,power=2,neighborhood_size=3,min_missing_support=2)
    assert np.allclose(a.sum(2),1) and np.allclose(b.sum(2),1)
    assert not np.allclose(a,b)
