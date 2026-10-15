from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import os
import hashlib
import json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ORIGINAL = Path(os.environ.get('CREDIT_MVP_CORE_RUN', PROJECT / 'runs/homecredit_temporal_v1_20261014')).resolve()
RAW = HERE / 'raw'
DERIVED = HERE / 'derived'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def protected_hashes():
    f = read(ORIGINAL / 'FROZEN_MANIFEST.json')
    for p, expected in f['files'].items():
        assert sha(p) == expected, f'Original frozen artifact changed: {p}'
    for p, expected in f['code'].items():
        assert sha(PROJECT / p) == expected, f'Original frozen code changed: {p}'
    return {str(p): sha(p) for p in [ORIGINAL / n for n in
            ('FROZEN_MANIFEST.json', 'PROTOCOL_LOCK.json', 'TEST_EVALUATION.json',
             'predictions_D.parquet', 'metrics.csv', 'budget.json', 'events.jsonl')]}


def source_hashes():
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py'))}
