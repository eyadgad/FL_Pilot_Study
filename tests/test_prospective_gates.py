import json,hashlib
from pathlib import Path
from scripts.aggregate_apgf_prospective import verify,NEED
from scripts.run_apgf_prospective import frozen_configs

ROOT=Path(__file__).resolve().parents[1]

def test_config_frozen_hashes():
    for name,want in frozen_configs().items():
        assert hashlib.sha256((ROOT/'configs'/name).read_bytes()).hexdigest()==want

def test_no_prospective_results_are_fabricated(tmp_path):
    result,rows=verify(tmp_path)
    assert result['status']=='INCOMPLETE' and result['observed_seeds']==0
    assert result['expected_seeds']==30 and result['errors']
    assert not rows

def test_strong_controls_are_mandatory():
    assert 'private_field_48' in NEED and 'fixed_peer_0p2_full48' in NEED

def test_all_pilot_records_are_genuine_development_scope():
    d=list((ROOT/'outputs').glob('real_mnist_*_summary.json'))
    assert len(d)==4
    for p in d:
        r=json.loads(p.read_text())
        assert r['dataset_provenance'].startswith('genuine CSV-origin MNIST')
        assert r['full10k_accuracy']>=.95
        assert len(r['selected_gates'])==3
        assert r['n_teacher_fit_per_client']==36
        assert r['n_teacher_validation_per_client']==12
        assert r['n_heldout_eval_per_client']==48
        for k in ('private_field_48','fixed_peer_0p2_full48','adaptive_peer'):
            assert r['methods'][k]['n_eval']==144
