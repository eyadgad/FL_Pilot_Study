from pathlib import Path
import numpy as np
from ucpa_fl.config import load_config
from ucpa_fl.datasets import load_data


def _idx(ds):
    return set(map(int, np.asarray(ds.indices).tolist()))


def test_five_client_splits_are_disjoint():
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root/'configs'/'sanity.yaml')
    b=load_data(cfg,1699)
    for cid in range(cfg.federation.n_clients):
        groups=[b.task_clients[cid],b.surrogate_clients[cid],b.artifact_clients[cid],b.calibration_clients[cid],b.eval_clients[cid]]
        sets=[_idx(x) for x in groups]
        for i in range(len(sets)):
            for j in range(i+1,len(sets)):
                assert sets[i].isdisjoint(sets[j])
        assert len(sets[3]) >= cfg.alignment.cfba_min_calibration_samples
        assert len(sets[4]) > 0
