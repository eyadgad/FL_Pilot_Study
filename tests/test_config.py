from pathlib import Path
import pytest, yaml
from ucpa_fl.config import load_config

def test_unknown_config_field_rejected(tmp_path):
    p=tmp_path/'x.yaml'; p.write_text('dataset:\n  name: synthetic\n  impossible: 1\n')
    with pytest.raises(ValueError): load_config(p)
