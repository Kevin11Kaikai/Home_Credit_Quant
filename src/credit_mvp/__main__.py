from __future__ import annotations
import argparse
from pathlib import Path

from .common import config_load, read_json, write_json


def main():
    parser=argparse.ArgumentParser(description='Frozen temporal credit-event MVP; all empirical results local.')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['smoke','run']:
        command=sub.add_parser(name)
        command.add_argument('--config',required=True)
    for name in ['report','absbox']:
        command=sub.add_parser(name)
        command.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    if args.command=='smoke':
        from .pipeline import smoke
        result=smoke(config_load(args.config))
        print(result['status'],result.get('source'))
        return 0 if result['status']=='PASS' else 1
    if args.command=='run':
        from .pipeline import run
        result=run(config_load(args.config))
        print(result['core_status'],result['extension_status'])
        return 0 if result['core_status']=='CORE_COMPLETE' else 2
    if args.command=='absbox':
        from .absbox_demo import run_absbox
        directory=Path(args.run_dir).resolve()
        cfg=read_json(directory/'config.json')
        result=run_absbox(cfg['absbox'],directory/'absbox')
        acceptance=read_json(directory/'acceptance.json')
        acceptance['extension_status']=result['status']
        acceptance['extension_reason']=result.get('reason')
        write_json(directory/'acceptance.json',acceptance)
        from .pipeline import refresh_acceptance_details
        refresh_acceptance_details(directory)
        print(result['status'])
    from .reporting import report
    print(report(Path(args.run_dir)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
