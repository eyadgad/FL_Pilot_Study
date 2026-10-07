import numpy as np
from ucpa_fl.alignment import rank_preserving_projection,pava_nonincreasing

def test_pava_nonincreasing_and_sum_preserved():
    y=np.array([.1,.4,.2,.3,0.0])
    q=pava_nonincreasing(y)
    assert np.all(q[:-1]>=q[1:]-1e-15)
    assert abs(q.sum()-y.sum())<1e-12

def test_rpga_preserves_complete_local_order():
    rng=np.random.default_rng(77)
    for _ in range(20):
        local=rng.random(200);local/=local.sum()
        prior=rng.random(200);prior/=prior.sum()
        out=rank_preserving_projection(local,prior)
        assert np.array_equal(np.argsort(-local),np.argsort(-out))
        assert np.all(out>=0) and abs(out.sum()-1)<1e-12

def test_rpga_is_prior_when_prior_respects_local_rank():
    local=np.array([.5,.3,.15,.05])
    prior=np.array([.4,.35,.2,.05])
    out=rank_preserving_projection(local,prior)
    assert np.allclose(out,prior,atol=1e-10)

def test_rpga_changes_incompatible_prior_but_keeps_rank():
    local=np.array([.6,.25,.1,.05])
    prior=np.array([.05,.1,.25,.6])
    out=rank_preserving_projection(local,prior)
    assert not np.allclose(out,prior)
    assert np.array_equal(np.argsort(-local),np.argsort(-out))
