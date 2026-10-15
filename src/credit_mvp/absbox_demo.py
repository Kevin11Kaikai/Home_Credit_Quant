from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import time
from urllib.parse import urlparse
import urllib.request

import numpy as np
import pandas as pd

from .common import Blocked, immutable_json, now, read_json, sha256, write_json

IMAGE='yellowbean/hastructure@sha256:aae12df53bb54f281dc93359297d617dd78c203ccd36c1f2c90e33b47f25438c'


def annual_to_monthly(rate):
    """Tested reference only; NOT used when passing annual CDR/CPR to AbsBox."""
    if not 0<=rate<=1:
        raise ValueError('Annual probability outside [0,1]')
    return -np.expm1(np.log1p(-rate)/12) if rate<1 else 1.0


def validate_endpoint(cfg):
    if not cfg.get('synthetic_only'):
        raise Blocked('Synthetic-only guard required')
    parsed=urlparse(cfg['engine_url'])
    if parsed.scheme not in ['http','https'] or parsed.username or parsed.password:
        raise Blocked('Invalid engine URL')
    if parsed.hostname not in ['127.0.0.1','localhost','::1']:
        # Current MVP intentionally provides no remote execution path.
        raise Blocked('Remote engine not enabled in this MVP; localhost only')


def reconcile(parsed,deal,atol=.05):
    accounts=parsed['accounts']
    bonds=parsed['bonds']
    pools=parsed['pool']['flow']
    flows=list(pools.values()) if isinstance(pools,dict) else [pools]
    assets_initial=sum(asset[2]['currentBalance'] for asset in deal.pool['assets'])
    bonds_initial=sum(spec['balance'] for _,spec in deal.bonds)
    initial_cash=sum(spec['balance'] for _,spec in deal.accounts)
    collected=sum(float(frame[[c for c in ['Principal','Interest','Prepayment','Recovery'] if c in frame]].sum().sum()) for frame in flows)
    bond_cash=sum(float(frame['cash'].sum()) for frame in bonds.values())
    end_cash=sum(float(frame['balance'].iloc[-1]) if len(frame) else 0 for frame in accounts.values())
    account_net=sum(float(frame['change'].sum()) for frame in accounts.values())
    checks={'initial_assets':float(assets_initial),'initial_bonds':float(bonds_initial),
            'initial_overcollateralization':float(assets_initial-bonds_initial),'initial_cash':float(initial_cash),
            'pool_collections':collected,'bond_cash':bond_cash,'ending_account_cash':end_cash,
            'cash_conservation_residual':initial_cash+collected-bond_cash-end_cash,
            'account_ledger_residual':initial_cash+account_net-end_cash,
            'tolerance':atol,'source':'REAL_ABSBOX_ENGINE_RESPONSE'}
    # No fees in the fixed official test01 waterfall.
    checks['passed']=abs(checks['cash_conservation_residual'])<=atol and abs(checks['account_ledger_residual'])<=atol and assets_initial==2200 and bonds_initial==2000
    return checks


def summarize_bonds(parsed,scenario,closing_date):
    rows=[]
    for name,frame in parsed['bonds'].items():
        if frame.empty:
            raise Blocked(f'Empty real engine bond flow: {name}')
        principal=float(frame.principal.sum())
        days=(pd.to_datetime(frame.index)-pd.Timestamp(closing_date)).days.to_numpy()
        wal=float(np.sum(days/365.25*frame.principal.to_numpy())/principal) if principal>0 else None
        rows.append({'scenario':scenario,'tranche':name,'principal_repaid':principal,
                     'interest_paid':float(frame.interest.sum()),'ending_balance':float(frame.balance.iloc[-1]),
                     'max_interest_due':float(frame.intDue.max()) if 'intDue' in frame else None,
                     'ending_interest_due':float(frame.intDue.iloc[-1]) if 'intDue' in frame else None,
                     'wal_years_repaid_principal':wal,
                     'wal_definition':'time from closing weighted by actual principal repayments; conditional on repaid principal, not unpaid balance',
                     'source':'REAL_ABSBOX_ENGINE_RESPONSE'})
    return rows


def run_absbox(cfg,directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    status_file=directory/'status.json'
    if status_file.exists() and read_json(status_file)['status']=='EXTENSION_COMPLETE':
        state=read_json(status_file)
        for filename,expected in state['response_hashes'].items():
            if sha256(directory/filename)!=expected:
                raise Blocked('AbsBox cached response changed')
        return state
    budget_file=directory/'attempt_budget.json'
    budget=read_json(budget_file) if budget_file.exists() else {'active_seconds':0,'attempts':0}
    start=time.monotonic()
    if budget['active_seconds']>=cfg['time_limit_seconds'] or budget['attempts']>=3:
        result={'status':'EXTENSION_BLOCKED','reason':'BLOCKED_DEPENDENCY: bounded extension attempt budget exhausted','actual_engine_run':False}
        write_json(status_file,result)
        return result
    budget['attempts']+=1
    try:
        validate_endpoint(cfg)
        immutable_json(directory/'config.json',{'synthetic_only':True,'engine_url':cfg['engine_url'],'image':IMAGE,'scenarios':cfg['scenarios'],
                       'official_deal':'absbox.examples.baseCase.test01','real_credit_data_transmitted':False,
                       'annual_rates_passed_directly':True,'recovery_lag_unit':'months (Monthly asset periods)'})
        from absbox import API
        from absbox.examples.baseCase import test01
        client_version=importlib.metadata.version('absbox')
        write_json(directory/'synthetic_deal.json',test01.json)
        with urllib.request.urlopen(cfg['engine_url'].rstrip('/')+'/version',timeout=5) as response:
            version=json.load(response)
        write_json(directory/'versions.json',{'client':client_version,'engine':version,'image':IMAGE})
        api=API(cfg['engine_url'],lang='english',check=True)
        rows=[]
        hashes={}
        checks={}
        scenarios={'ZeroPressure':{'annual_cdr':0.0,'annual_cpr':0.0,'recovery':0.6,'recovery_lag_months':3},**cfg['scenarios']}
        for name,s in scenarios.items():
            if time.monotonic()-start+budget['active_seconds']>=cfg['time_limit_seconds']:
                raise Blocked('Extension runtime budget exhausted')
            assump=('Pool',('Mortgage',{'CDR':s['annual_cdr']},{'CPR':s['annual_cpr']},{'Rate':s['recovery'],'Lag':s['recovery_lag_months']},None),None,None)
            req=api.run(test01,poolAssump=assump,runAssump=[],debug=True)
            # Only the explicitly defined synthetic example enters the engine.
            write_json(directory/f'{name}_request.json',json.loads(req) if isinstance(req,str) else req)
            response_path=directory/f'{name}_response.json'
            if response_path.exists():
                raw=read_json(response_path)
            else:
                raw=api.run(test01,poolAssump=assump,runAssump=[],read=False)
                if raw is None:
                    raise Blocked('Engine returned no response')
                write_json(response_path,raw)
            parsed=test01.read(raw)
            hashes[response_path.name]=sha256(response_path)
            checks[name]=reconcile(parsed,test01)
            for bond,frame in parsed['bonds'].items():
                frame.to_csv(directory/f'{name}_bond_{bond}.csv')
            for account,frame in parsed['accounts'].items():
                frame.to_csv(directory/f'{name}_account_{account}.csv')
            flow=parsed['pool']['flow']
            if isinstance(flow,dict):
                for pool,frame in flow.items():
                    frame.to_csv(directory/f'{name}_pool_{pool}.csv')
            else:
                flow.to_csv(directory/f'{name}_pool.csv')
            rows.extend(summarize_bonds(parsed,name,test01.dates['closing']))
        pd.DataFrame(rows).to_csv(directory/'tranche_summary.csv',index=False)
        write_json(directory/'reconciliation.json',checks)
        if not all(check['passed'] for check in checks.values()):
            raise Blocked('Real engine returned data but cash reconciliation failed; see reconciliation.json')
        result={'status':'EXTENSION_COMPLETE','actual_engine_run':True,'mock':False,'client_version':client_version,
                'engine_version':version,'response_hashes':hashes,'reconciliation_passed':True,'utc':now(),
                'scope':'Independent synthetic waterfall mechanics; not Home Credit annual CDR, CLO valuation or investment backtest'}
    except Exception as error:
        import traceback
        result={'status':'EXTENSION_BLOCKED','reason':'BLOCKED_DEPENDENCY: '+str(error),'actual_engine_run':bool(list(directory.glob('*_response.json'))),
                'mock':False,'traceback':traceback.format_exc(),'utc':now()}
    budget['active_seconds']+=time.monotonic()-start
    write_json(budget_file,budget)
    write_json(status_file,result)
    return result
