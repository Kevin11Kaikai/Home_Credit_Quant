from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from datetime import datetime, timezone

import numpy as np
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'logs'/'matplotlib'))


class Blocked(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, default=_json_default)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(array):
    x = np.ascontiguousarray(array)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if canonical(read_json(path)) != canonical(value):
            raise Blocked(f'Immutable record mismatch: {path}')
    else:
        write_json(path, value)


def config_load(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    # All relative paths are project-root relative, not invocation-directory relative.
    return cfg


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def source_hashes():
    paths = sorted((ROOT / 'src').rglob('*.py'))
    paths += [ROOT / 'pyproject.toml', ROOT / 'SOURCE_MANIFEST.json']
    if (ROOT / 'requirements.lock').exists():
        paths.append(ROOT / 'requirements.lock')
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in paths}


def freeze(path, files, metadata):
    path = Path(path)
    payload = {'files': {str(Path(f).resolve()): sha256(f) for f in sorted(files)},
               'metadata': metadata, 'code': source_hashes()}
    immutable_json(path, payload)
    return payload


def verify_freeze(path):
    frozen = read_json(path)
    for f, expected in frozen['files'].items():
        if not Path(f).is_file() or sha256(f) != expected:
            raise Blocked(f'Frozen artifact changed: {f}')
    if source_hashes() != frozen['code']:
        raise Blocked('Code/dependency/source changed after freeze; explicit correction record and a new version required. D remains seen if previously evaluated.')
    for f, expected in frozen.get('metadata',{}).get('data_files',{}).items():
        if not Path(f).is_file() or sha256(f) != expected:
            raise Blocked(f'Frozen input data changed: {f}')
    return frozen


def hardware():
    import torch
    v = psutil.virtual_memory()
    disk = psutil.disk_usage(str(ROOT))
    return {'utc': now(), 'python': platform.python_version(), 'platform': platform.platform(),
            'cpu_count': os.cpu_count(), 'ram_total_bytes': v.total, 'ram_available_bytes': v.available,
            'disk_free_bytes': disk.free, 'cuda_available': torch.cuda.is_available(),
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'vram_bytes': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            'tracking': 'local JSON/CSV (no existing workspace MLflow configuration)'}


def event(run_dir, stage, status, **extra):
    record = {'utc': now(), 'stage': stage, 'status': status, **extra}
    path = Path(run_dir) / 'events.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(canonical(record) + '\n')
    print(f'[{stage}] {status}', flush=True)


class Budget:
    """Durable cumulative per-family wall time, including interrupted attempts."""
    def __init__(self, path, limits):
        self.path = Path(path)
        self.limits = limits
        self.state = read_json(path) if self.path.exists() else {'used': {}, 'trials': {}, 'active': None}
        self.state.setdefault('trials',{})
        if self.state['active']:
            active = self.state['active']
            elapsed = min(time.time() - active['started'], active['allowance'])
            self.state['used'][active['family']] = self.used(active['family']) + max(0, elapsed)
            trial=active.get('trial_id',active['family'])
            self.state['trials'][trial]=self.state['trials'].get(trial,0)+max(0,elapsed)
            self.state['active'] = None
            write_json(self.path, self.state)

    def used(self, family):
        return self.state['used'].get(family, 0.0)

    def remaining(self, family):
        return max(0.0, self.limits[family] - self.used(family))

    def begin(self, family, cap, trial_id=None):
        trial_id=trial_id or family
        allowance = min(cap-self.state['trials'].get(trial_id,0), self.remaining(family))
        if allowance <= 0:
            raise Blocked(f'{family} cumulative budget exhausted')
        self.state['active'] = {'family': family, 'trial_id':trial_id, 'started': time.time(), 'allowance': allowance}
        write_json(self.path, self.state)
        return allowance

    def finish(self):
        active = self.state['active']
        if active:
            elapsed = time.time() - active['started']
            self.state['used'][active['family']] = self.used(active['family']) + elapsed
            trial=active['trial_id']
            self.state['trials'][trial]=self.state['trials'].get(trial,0)+elapsed
            self.state['active'] = None
            write_json(self.path, self.state)
            return elapsed
