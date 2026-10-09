import numpy as np
import pytest
from src.apgf import quantized_teacher_stats,candidate_fields,choose_weight,fit_gated_fields

def record():
    x=np.ones(784)*.5
    y=np.ones(784)/784
    return {'x':x,'oracle_map':y,'pred':0}

def test_private_is_exact_fallback():
    a=[[record() for _ in range(8)] for _ in range(3)]
    o=fit_gated_fields(a,a)
    np.testing.assert_allclose(o['private'],o['selected'])
    assert [d['selected_weight'] for d in o['decisions']]==[0.]*3

def test_packets_and_counts():
    p=quantized_teacher_stats([record()]*6)
    assert len(p[0])==len(p[1])==790
    c,ledger=candidate_fields([p,p,p])
    assert ledger['per_client_downlink_bytes']==5*786
    assert ledger['candidate_fields_sent_per_client']==5
    assert all(len(v)==5 for v in c)

def test_fail_closed_on_broken_records():
    with pytest.raises(ValueError):quantized_teacher_stats([])
    with pytest.raises(ValueError):choose_weight({0:np.ones(784)},[record()]*3)

def test_peer_excludes_self():
    x=record();y=record();y['oracle_map']=np.linspace(.001,.01,784)
    a=quantized_teacher_stats([x]*5);b=quantized_teacher_stats([y]*5)
    f,_=candidate_fields([a,b])
    np.testing.assert_allclose(f[0][0],candidate_fields([a,a])[0][0][0.0])
    assert not np.allclose(f[0][.2],f[0][0])
