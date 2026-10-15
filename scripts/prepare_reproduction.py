"""Create a new run configuration. Never downloads, trains, or resumes."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='homecredit_reproduction_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--output', default='configs/reproduction.yaml')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('run-id must be a simple directory name')
    target = (ROOT/args.output).resolve()
    if ROOT not in target.parents or target.exists() or (ROOT/'runs'/args.run_id).exists():
        parser.error('Refusing to overwrite or write outside this checkout')
    config = yaml.safe_load((ROOT/'configs/mvp.yaml').read_text(encoding='utf-8'))
    config['run_id'] = args.run_id
    config['protocol_id'] = 'homecredit_temporal_v1_reproduction'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    print(f'Created {target.relative_to(ROOT)}; no experiment executed.')
    print(f'Run directory will be runs/{args.run_id}')

if __name__=='__main__':main()
