"""Protocol/codec tests; no synthetic-image research smoke."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from study_v7.data import verify_data
from study_v7.codec import stat_encode,stat_decode,field_encode,field_decode
from study_v7.correction import SpatialCorrection
from qfscad_impl.core import serialize_state,deserialize_state

def test_frozen_hashes():
    expected=json.loads((ROOT/'configs'/'FROZEN_CONFIG_HASHES.json').read_text())
    assert len(expected)==12
    for f,digest in expected.items():assert hashlib.sha256((ROOT/'configs'/f).read_bytes()).hexdigest()==digest

def test_60_new_training_seeds():
    p=json.loads((ROOT/'configs'/'PROTOCOL.json').read_text())
    assert len(p['seed_matrix'])==12
    assert all(len(seeds)==5 for seeds in p['seed_matrix'].values())
    assert len(set(sum(p['seed_matrix'].values(),[])))==60
    assert p['teachers_per_client']==48 and p['evaluation_per_client']==64

def test_original_77_parameters():
    net=SpatialCorrection()
    assert sum(t.numel() for t in net.parameters())==77
    pkt=serialize_state(net.state_dict())
    assert len(pkt)==93
    assert len(deserialize_state(pkt,net.state_dict()))==4
    with pytest.raises(ValueError):deserialize_state(pkt[:-1],net.state_dict())

def test_quantized_packet_accounting_with_offline_numeric_vectors():
    # Codec unit test: deterministic numbers, NOT experimental image data.
    x=np.arange(784,dtype=float)+1
    y,n=stat_decode(stat_encode(x,48),784)
    assert n==48 and len(stat_encode(x,48))==790
    field=field_decode(field_encode(x),784)
    assert len(field)==784 and np.isfinite(field).all()
    with pytest.raises(ValueError):field_decode(field_encode(x),1024)

def test_no_mnist_fallback():
    with pytest.raises(FileNotFoundError):verify_data('mnist',ROOT/'this_directory_does_not_exist')

def test_no_cifar_fallback():
    with pytest.raises(FileNotFoundError):verify_data('cifar10',ROOT/'this_directory_does_not_exist')

def test_no_synthetic_dataset():
    with pytest.raises(ValueError):verify_data('synthetic',ROOT)

def test_corrected_cifar_shapes_match_protocol():
    proto=json.loads((ROOT/'configs'/'PROTOCOL.json').read_text())
    assert proto['datasets']['cifar10']['shape']==[3,32,32]
    assert proto['topk_spatial']['cifar10']<=1024
    assert proto['require_nonregression_in_edi_and_functional']
