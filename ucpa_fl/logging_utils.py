from __future__ import annotations
from pathlib import Path
import json, hashlib, os, time, uuid
from typing import Any


def _jsonable(x: Any):
    if hasattr(x, 'item'):
        try: return x.item()
        except Exception: pass
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_jsonable)
        f.write('\n')
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, default=_jsonable) + '\n'
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class RunLogger:
    def __init__(self, output_root: str | Path, experiment: str, seed: int, config: dict, env: dict):
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        rid = f'{experiment}__seed{seed}__{timestamp}__{uuid.uuid4().hex[:8]}'
        self.run_dir = Path(output_root) / experiment / rid
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / 'checkpoints').mkdir()
        (self.run_dir / 'artifacts').mkdir()
        self.events = self.run_dir / 'events.jsonl'
        self.metrics = self.run_dir / 'metrics.jsonl'
        cfg_blob=json.dumps(config,sort_keys=True,default=_jsonable).encode('utf-8')
        self.manifest = {
            'run_id': rid,
            'config_sha256': hashlib.sha256(cfg_blob).hexdigest(),
            'experiment': experiment,
            'seed': seed,
            'created_unix': time.time(),
            'config': config,
            'environment': env,
            'status': 'running',
        }
        atomic_json(self.run_dir / 'manifest.json', self.manifest)
        self.event('run_started')

    def event(self, event: str, **payload):
        append_jsonl(self.events, {'time': time.time(), 'event': event, **payload})

    def metric(self, name: str, value: float, **context):
        append_jsonl(self.metrics, {'time': time.time(), 'metric': name, 'value': float(value), **context})

    def save_json(self, rel: str, payload: dict):
        atomic_json(self.run_dir / rel, payload)

    def finalize(self, status: str = 'completed', **extra):
        self.event('run_finalized', status=status)
        self.manifest['status'] = status
        self.manifest['finished_unix'] = time.time()
        self.manifest.update(extra)
        atomic_json(self.run_dir / 'manifest.json', self.manifest)
        files = []
        for p in sorted(self.run_dir.rglob('*')):
            if p.is_file() and p.name != 'integrity.sha256':
                files.append((str(p.relative_to(self.run_dir)), sha256_file(p)))
        with open(self.run_dir / 'integrity.sha256', 'w', encoding='utf-8') as f:
            for rel, digest in files:
                f.write(f'{digest}  {rel}\n')
