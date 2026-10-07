import numpy as np
from ucpa_fl.rwssa import learn_client_gain,global_safety,safety_weight,weight_map,select_scale
from ucpa_fl.datasets import dirichlet_partition

def test_positive_risk_gain_and_freeze_rho():
    gains=np.array([[[.2,.5,-.2]],[[.4,.1,-.3]]])
    A=safety_weight(gains,0.30)
    assert np.all((A>=.3)&(A<=1.0))
    assert abs(A[0,2]-.30)<1e-10
    assert A[0,0]>.30
    local=np.array([.9,.1,0.]);p=np.array([.1,.8,.1]);out=weight_map(local,p,A[0],.5)
    assert np.isclose(out.sum(),1.)
    assert not np.allclose(out,local)

def test_missing_classes_do_not_fake_fidelity_gains():
    gains=np.full((2,3,5),np.nan)
    assert np.allclose(global_safety(gains),0)
    assert np.allclose(safety_weight(gains,0.3),0.3)

def test_calibration_requires_independent_records_and_default_is_safe():
    C,D=2,4;prior=np.ones((C,D))/D;A=np.ones((C,D))
    cert=select_scale([],prior,A)
    assert cert['lambda']==0 and not cert['certified']

def test_dirichlet_no_silent_iid_repair():
    labels=np.repeat(np.arange(5),200)
    groups=dirichlet_partition(labels,8,0.1,42,min_samples=55)
    joined=np.concatenate(groups)
    assert len(np.unique(joined))==len(labels)
    assert joined.size==len(labels)
    assert min(len(x) for x in groups)>=55
    hist=np.stack([np.bincount(labels[g],minlength=5)/len(g) for g in groups])
    assert float(hist.std(axis=0).mean())>.05

def test_sparse_gain_encoding_and_budget():
    from ucpa_fl.rwssa import sparse_quantized_client_gains
    a=np.tile(np.array([[[.1,.2,-.5,0]]]),(3,1,1))
    mask=np.tile(np.array([[[True,False,True,False]]]),(3,1,1))
    b,n=sparse_quantized_client_gains(a,mask)
    assert n==6
    assert np.isnan(b[:,0,1]).all()
    assert np.allclose(b[:,0,2],-.5)
    assert np.nanmax(np.abs(b-a))<.004
