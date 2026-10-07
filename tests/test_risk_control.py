import numpy as np
import torch

from ucpa_fl.risk_control import (
    monotone_envelope_nondec,
    crc_upper_empirical_risk,
    choose_largest_certified_beta,
    perturbation_auc_many,
)


def test_monotone_envelope_nondec():
    x=np.array([[0.0,0.2,0.1,0.3],[0.0,0.05,0.08,0.07]])
    y=monotone_envelope_nondec(x)
    assert np.all(np.diff(y,axis=1)>=-1e-15)
    np.testing.assert_allclose(y[0],[0,.2,.2,.3])
    np.testing.assert_allclose(y[1],[0,.05,.08,.08])


def test_crc_formula_and_beta_selection():
    grid=[0,.1,.2,.4]
    # n=39 => correction 1/40=.025. First three actions remain below alpha=.05.
    L=np.tile(np.array([0.0,.01,.02,.08]),(39,1))
    out=choose_largest_certified_beta(L,grid,alpha=.05)
    assert out['certified'] is True
    assert out['beta']==.2
    upper=np.asarray(out['upper_risk'])
    assert upper[2] <= .05 < upper[3]
    expected=(39/40)*np.array([0,.01,.02,.08])+1/40
    np.testing.assert_allclose(upper,expected)


def test_crc_fallback_not_falsely_certified_with_tiny_n():
    out=choose_largest_certified_beta(np.zeros((2,2)),[0,.1],alpha=.05)
    assert out['beta']==0
    assert out['certified'] is False


def test_perturbation_auc_many_bounded():
    class M(torch.nn.Module):
        def forward(self,x):
            z=x.flatten(1).sum(1)
            return torch.stack([z,-z],dim=1)
    model=M().eval()
    x=torch.ones(1,4,4)
    maps=np.stack([np.linspace(1,0,16),np.linspace(0,1,16)])
    d,i=perturbation_auc_many(model,x,0,maps,torch.device('cpu'),steps=4)
    assert np.all((d>=0)&(d<=1))
    assert np.all((i>=0)&(i<=1))


def test_crc_monte_carlo_expected_risk_sanity():
    # Engineering sanity, not a proof. Standard CRC controls the *expected*
    # risk of the selected rule over calibration randomness; it is not a
    # high-probability statement that every selected rule has true risk <=alpha.
    rng=np.random.default_rng(7)
    true=np.array([0.0,.005,.015,.08])
    grid=[0,.1,.2,.4]
    alpha=.05
    selected_true=[]
    reps=4000
    n=64
    for _ in range(reps):
        L=(rng.random((n,len(true))) < true).astype(float)
        # enforce the nested/monotone loss family required by CRC
        L=np.maximum.accumulate(L,axis=1)
        out=choose_largest_certified_beta(L,grid,alpha)
        selected_true.append(true[out['index']])
    assert float(np.mean(selected_true)) <= alpha + .005
