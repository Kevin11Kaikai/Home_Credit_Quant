"""Verify the published bytes and aggregate statuses without data or training."""
from pathlib import Path
import argparse, csv, hashlib, json

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',action='store_true');args=p.parse_args()
    manifest=read('RELEASE_MANIFEST.json')
    for name,expected in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise SystemExit('Publication hash mismatch: '+name)
    status=read('STATUS.json')
    assert status['core']=='CORE_COMPLETE' and status['absbox']=='EXTENSION_COMPLETE'
    assert status['followup']=='PAUSED_BY_USER'
    assert status['s6e7_completed_members']==0 and status['automatic_resume_allowed'] is False
    with (ROOT/'evidence/core/metrics.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==13 and {r['n'] for r in rows}=={'56001'}
    for model in ['LightGBM','TabM']:
        pair={r['variant']:r for r in rows if r['model']==model and r['seed']=='42'}
        assert float(pair['raw']['roc_auc'])==float(pair['calibrated']['roc_auc'])
        assert float(pair['calibrated']['log_loss'])>float(pair['raw']['log_loss'])
    print(f'PASS: {len(manifest["files"])} publication files verified; 13 historical core metric rows; paused follow-up preserved.')
    if args.summary:print(json.dumps(status,indent=2))
    print('This verifies published bytes and aggregate consistency, not new inference or independent replication.')

if __name__=='__main__':main()
