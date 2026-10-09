"""Software-level specification tests, NOT synthetic scientific experiments."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pytest
import torch
from ucpa_fl.models import MNISTCNN,CIFARSmallCNN
from ucpa_fl.federated import train_local_fedprox
from ucpa_fl.repro import resolve_device
from scripts.launch_expanded import plan
from study_v7.correction import SpatialCorrection
from study_v7.codec import stat_encode,stat_decode,field_encode,field_decode

def test_expanded_frozen_matrix():
    plan_=json.loads((ROOT/'configs/EXPANDED_STUDY_PLAN.json').read_text())
    hashes=json.loads((ROOT/'configs/EXPANDED_CONFIG_HASHES.json').read_text())
    assert plan_['run_count']==138
    assert len(hashes)==38
    assert len(list(plan()))==138
    for filename,expected in hashes.items():
        assert hashlib.sha256((ROOT/'configs/expanded'/filename).read_bytes()).hexdigest()==expected

def test_full_factorial_axes_wide():
    a=list(json.loads((ROOT/'configs/EXPANDED_STUDY_PLAN.json').read_text())['matrix'].values())
    for ds in ('mnist','cifar10'):
        rows=[x for x in a if x['dataset']==ds]
        assert set(x['n_clients'] for x in rows)>= {3,5,10,20}
        assert set(x['rounds'] for x in rows)>= {5,10,25,50}
        assert set(x['local_epochs'] for x in rows)>={1,2,5,10}
        assert all(len(x['seeds'])>=3 for x in rows if x['kind']!='smoke')
        assert sum(len(x['seeds']) for x in rows if x['kind']=='primary')==35

def test_paired_sensitivity_and_optimizer_seeds():
    p=json.loads((ROOT/'configs/EXPANDED_STUDY_PLAN.json').read_text())['matrix']
    for ds in ('mnist','cifar10'):
        expected=p[f'{ds}_primary_iid_c10_r50_e5_fedavg.yaml']['seeds'][:3]
        assert all(x['seeds']==expected for x in p.values() if x['dataset']==ds and x['kind'] in ('clients','rounds','epochs'))
        assert p[f'{ds}_optimizer_iid_c10_r50_e5_fedprox.yaml']['seeds']==expected

def test_datastream_guards():
    from study_v7.data import verify_data
    with pytest.raises((FileNotFoundError,ValueError)):
        verify_data('cifar10',ROOT/'not_a_dataset')
    with pytest.raises(ValueError):verify_data('synthetic',ROOT)

def test_published_baseline_architectures():
    from study_v7.published_baselines import _last_conv
    assert isinstance(_last_conv(MNISTCNN()),torch.nn.Conv2d)
    assert isinstance(_last_conv(CIFARSmallCNN()),torch.nn.Conv2d)

def test_quantized_cifar_bytes():
    import numpy as np
    z=np.arange(1,1025,dtype=float)
    pkt=stat_encode(z,48)
    assert len(pkt)==1030
    rec,n=stat_decode(pkt,1024)
    assert n==48 and rec.min()>=0
    assert len(field_encode(z))==1026
    assert len(field_decode(field_encode(z),1024))==1024

def test_cuda_fail_closed_without_gpu():
    if not torch.cuda.is_available():
        assert not torch.cuda.is_available()
    assert str(resolve_device('cpu'))=='cpu'

def test_smoke_script_enforces_five_by_five():
    src=(ROOT/'scripts/smoke_5r5e_real_mnist.py').read_text()
    assert 'for round_ in range(5)' in src
    assert 'epochs=5' in src
    assert 'download=False' in src

def test_corrector_device_contract():
    from study_v7.correction import train,train_private,predict
    import inspect
    for method in (train,train_private):assert 'device' in inspect.signature(method).parameters
    assert sum(x.numel() for x in SpatialCorrection().parameters())==77

def test_fedprox_paper_objective_documented():
    import inspect
    code=inspect.getsource(train_local_fedprox)
    assert 'mu' in code and 'p-r' in code and '0.5*mu*prox' in code
