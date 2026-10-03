import numpy as np
from ucpa import align


def test_rows_normalize_and_finite():
    E=np.array([[.7,.2,.1],[.6,.3,.1],[.1,.2,.7]])
    V=np.full_like(E,.002)
    A,d=align(E,V,return_diagnostics=True)
    assert np.all(np.isfinite(A))
    assert np.allclose(A.sum(1),1.0)
    assert 1.0 <= d['effective_peers_mean'] <= 3.0


def test_identical_explanations_are_fixed_point():
    E=np.tile(np.array([.6,.3,.1]),(4,1))
    V=np.full_like(E,.003)
    A=align(E,V)
    assert np.allclose(A,E,atol=1e-12)


def test_strongly_incompatible_peer_has_small_influence():
    E=np.array([[.99,.01],[.01,.99]])
    V=np.full_like(E,1e-8)
    A=align(E,V)
    assert A[0,0] > .98 and A[1,1] > .98
