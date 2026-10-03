from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import yaml

@dataclass
class DatasetConfig:
    name: str = 'synthetic'
    root: str = './data'
    download: bool = True
    n_classes: int = 10
    train_limit: int | None = None
    test_limit: int | None = None

@dataclass
class FederationConfig:
    n_clients: int = 8
    rounds: int = 15
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    participation_rate: float = 1.0
    partition: str = 'dirichlet'  # iid | dirichlet
    dirichlet_alpha: float = 0.1
    min_client_samples: int = 16
    sample_size_sigma: float = 0.0

@dataclass
class ShiftConfig:
    kind: str = 'none'  # none | rotation | patch | erasing | color | drift_rotation
    strength: float = 1.0
    rotation_max_deg: float = 25.0
    patch_size: int = 4
    patch_value: float = 1.0
    drift_start_round: int = 8
    drift_end_round: int = 15

@dataclass
class SurrogateConfig:
    kind: str = 'linear'  # linear | task_ig
    source: str = 'linear_surrogate'  # linear_surrogate | task_ig
    epochs: int = 1
    lr: float = 0.1
    temperature: float = 3.0
    l1: float = 1e-4
    hidden_dim: int = 128
    max_samples_per_class: int = 32
    eval_samples_per_class: int = 24
    ig_steps: int = 16
    artifact_samples_per_class: int = 24

@dataclass
class ArtifactConfig:
    topk: int = 128
    clip_radius: float = 5.0
    quant_bits: int = 8
    dp_sigma: float = 0.1
    missing_variance_scale: float = 4.0
    send_variance: bool = True
    sparse_intersection_only: bool = True

@dataclass
class AlignmentConfig:
    beta: float = 0.2
    methods: list[str] = field(default_factory=lambda: [
        'local', 'fedattr_mean', 'xfedalign_median', 'ucpa',
        'ucpa_whole_only', 'ucpa_coord_only', 'cluster'
    ])
    ucpa_z: float = 2.0
    ucpa_h: float = 0.048
    ucpa_variance_floor_fraction: float = 0.05
    cluster_k: int = 3

@dataclass
class EvaluationConfig:
    deletion_steps: int = 20
    eval_batch_size: int = 128
    max_eval_samples: int = 256
    compute_oracle_local_fidelity: bool = True
    oracle_samples_per_class: int = 64
    topk_overlap_k: int = 128
    bootstrap_samples: int = 2000
    deletion_insertion_samples_per_client: int = 24
    primary_metric: str = 'local_fidelity_jsd'

@dataclass
class AttackConfig:
    enabled: bool = False
    fraction: float = 0.25
    strength: float = 0.3
    kind: str = 'artifact_shift'  # artifact_shift | random_support

@dataclass
class LoggingConfig:
    output_root: str = './outputs'
    save_checkpoints: bool = True
    save_artifacts: bool = True
    log_every_round: bool = True

@dataclass
class ExperimentConfig:
    experiment_name: str = 'sanity'
    seeds: list[int] = field(default_factory=lambda: [1])
    device: str = 'auto'
    deterministic: bool = True
    num_workers: int = 0
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    shift: ShiftConfig = field(default_factory=ShiftConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(cls, payload: dict[str, Any] | None):
    payload = payload or {}
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(f'Unknown fields for {cls.__name__}: {sorted(unknown)}')
    return cls(**payload)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    top = dict(raw)
    top['dataset'] = _merge_dataclass(DatasetConfig, raw.get('dataset'))
    top['federation'] = _merge_dataclass(FederationConfig, raw.get('federation'))
    top['shift'] = _merge_dataclass(ShiftConfig, raw.get('shift'))
    top['surrogate'] = _merge_dataclass(SurrogateConfig, raw.get('surrogate'))
    top['artifact'] = _merge_dataclass(ArtifactConfig, raw.get('artifact'))
    top['alignment'] = _merge_dataclass(AlignmentConfig, raw.get('alignment'))
    top['evaluation'] = _merge_dataclass(EvaluationConfig, raw.get('evaluation'))
    top['attack'] = _merge_dataclass(AttackConfig, raw.get('attack'))
    top['logging'] = _merge_dataclass(LoggingConfig, raw.get('logging'))
    return _merge_dataclass(ExperimentConfig, top)
