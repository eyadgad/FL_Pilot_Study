from pathlib import Path
import hashlib,json
from scripts.aggregate_apgf_prospective import NEED
from scripts.run_apgf_prospective import frozen_configs
from src.apgf import candidate_fields,quantized_teacher_stats

def test_frozen_prospective_seed_count():
    root=Path(__file__).resolve().parents[1]
    z=json.loads((root/'configs'/'APGF_FROZEN_PROSPECTIVE.json').read_text())
    assert len(z['new_seeds'])==6
    assert len(set(sum(z['new_seeds'].values(),[])))==30
    assert sum(map(len,z['new_seeds'].values()))==30

def test_no_synthetic_science_loader():
    root=Path(__file__).resolve().parents[1]
    src=(root/'ucpa_fl'/'datasets.py').read_text()
    assert 'Synthetic data prohibited for APGF' in src
    assert 'def _synthetic(' not in src

def test_5_downlinks_charged_for_private_weight_selection():
    import numpy as np
    r={'x':np.ones(784)/2,'oracle_map':np.ones(784)/784, 'pred':0}
    packet=quantized_teacher_stats([r]*4)
    _,ledger=candidate_fields([packet,packet])
    assert ledger['per_client_downlink_bytes']==5*786
    assert ledger['per_client_uplink_bytes']==[1580,1580]
    assert ledger['candidate_fields_sent_per_client']==5
