from __future__ import annotations

import copy
import gc
from pathlib import Path
import subprocess
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd

from .calibration import PositiveSigmoid, numeric_ties
from .common import (ROOT, Blocked, Budget, array_hash, digest, event, freeze, hardware,
                     immutable_json, now, project_path, read_json, sha256, source_hashes,
                     verify_freeze, write_json)
from .data import Dataset, DevelopmentData, LIMITATIONS, Preprocessor, acquire, development
from .evaluation import metrics, moving_block_bootstrap, reliability, reliability_edges, weekly
from .models import fit_lightgbm, fit_tabm, make_bins, predict, select_best
from .portfolio import evaluate_portfolios


def synthetic_development():
    rng=np.random.default_rng(20261014)
    num=rng.normal(size=(768,12)).astype(np.float32)
    num[::11,2]=np.nan
    num[:,11]=1
    cat=rng.integers(0,4,size=(768,2)).astype(float)
    cat[::13,0]=np.nan
    cat[-80:,1]=999
    y=rng.binomial(1,1/(1+np.exp(-(np.nan_to_num(num[:,0])*.8-1)))).astype(np.int64)
    pp=Preprocessor().fit(num[:512],cat[:512])
    na,ca=pp.transform(num[:512],cat[:512])
    nb,cb=pp.transform(num[512:],cat[512:])
    return DevelopmentData(na,ca,y[:512],nb,cb,y[512:]),pp


def smoke(cfg):
    directory=ROOT/'runs'/'engineering_smoke_v1'
    directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/'smoke.json'
    if receipt.exists():
        old=read_json(receipt)
        if old.get('code')==source_hashes():
            return old
        # No repeated training: check changed code with unit tests, retain smoke evidence.
        return {**old,'current_code_matches':False,'note':'Existing one-per-model smoke preserved; rerun unit tests for later source edits.'}
    folder=project_path(cfg['data_dir'])
    # This fallback is explicitly engineering-only; it supplies no empirical results.
    if not (folder/'info.json').exists():
        dev,pp=synthetic_development()
        source='SYNTHETIC_ENGINEERING_ONLY_REAL_DATA_UNAVAILABLE'
    else:
        data=Dataset(folder)
        a=data.splits['A'][:2048]
        b=data.splits['B'][:1024]
        an,ac=data.features(a)
        bn,bc=data.features(b)
        pp=Preprocessor().fit(an,ac)
        an,ac=pp.transform(an,ac)
        bn,bc=pp.transform(bn,bc)
        dev=DevelopmentData(an,ac,data.labels('A')[:len(a)],bn,bc,data.labels('B')[:len(b)])
        source='A_B_DEVELOPMENT_SLICE'
    bins=make_bins(dev.num_A,cfg['tabm']['n_bins'])
    started=time.monotonic()
    results={}
    for family in ['LightGBM','TabM']:
        out=directory/family
        try:
            if family=='LightGBM':
                result=fit_lightgbm(dev,cfg['lightgbm']['candidates'][0],cfg,42,out,120,smoke=True)
            else:
                result=fit_tabm(dev,cfg['tabm']['candidates'][0],cfg,42,out,120,pp.cardinalities,bins,smoke=True)
            passed=result.get('eligible',False) and result.get('B') is not None and result.get('checkpoint') and Path(result['checkpoint']).exists()
            results[family]={'status':'ENGINEERING_SMOKE_PASS' if passed else 'INCOMPLETE','training':result,'empirical_credit_result':False}
        except Exception as e:
            results[family]={'status':'FAILED','error':str(e)}
    joblib.dump(pp,directory/'preprocessor.joblib')
    receipt_data={'status':'PASS' if all(r['status']=='ENGINEERING_SMOKE_PASS' for r in results.values()) else 'FAILED',
                  'source':source,'real_credit_training_executed':source=='A_B_DEVELOPMENT_SLICE',
                  'formal_training_executed':False,'seconds':time.monotonic()-started,'hardware':hardware(),
                  'results':results,'dimensions':pp.input_dimensions,'code':source_hashes()}
    write_json(receipt,receipt_data)
    return receipt_data


def train_and_select(data,pp,cfg,run_dir):
    dev=development(data,pp)
    if not np.isin(dev.y_A,[0,1]).all() or len(np.unique(dev.y_A))!=2:
        raise Blocked('A labels are not binary with both classes.')
    import torch
    cuda=torch.cuda.is_available()
    bins=make_bins(dev.num_A,cfg['tabm']['n_bins']) if cuda else None
    if bins is not None:
        joblib.dump(bins,run_dir/'bins.joblib')
    budget=Budget(run_dir/'budget.json',{'LightGBM':cfg['lightgbm']['total_seconds'],'TabM':cfg['tabm']['total_seconds']})
    # Charge the already executed smoke to the matching family budget once.
    if 'smoke_charged' not in budget.state:
        smoke_receipt=ROOT/'runs'/'engineering_smoke_v1'/'smoke.json'
        if smoke_receipt.exists():
            for family,record in read_json(smoke_receipt)['results'].items():
                if 'training' in record:
                    budget.state['used'][family]=budget.used(family)+record['training']['seconds']
        budget.state['smoke_charged']=True
        write_json(budget.path,budget.state)
    records={'LightGBM':[],'TabM':[]}
    failures=[]
    def fit(family,candidate,seed,phase):
        subsection='lightgbm' if family=='LightGBM' else 'tabm'
        directory=run_dir/'fits'/f'{family}_{candidate["id"]}_seed{seed}'
        if (directory/'result.json').exists():
            return read_json(directory/'result.json')
        if (directory/'failure.json').exists():
            return None
        if family=='TabM' and not cuda:
            failures.append({'family':family,'reason':'CUDA_UNAVAILABLE'})
            return None
        if budget.remaining(family)<=0:
            failures.append({'family':family,'reason':'TOTAL_BUDGET_EXHAUSTED','config':candidate,'seed':seed})
            return None
        trial_id=f'{family}_{candidate["id"]}_seed{seed}'
        if budget.state['trials'].get(trial_id,0)>=cfg[subsection]['fit_seconds']:
            failures.append({'family':family,'reason':'TRIAL_BUDGET_EXHAUSTED','trial_id':trial_id})
            return None
        allowed=budget.begin(family,cfg[subsection]['fit_seconds'],trial_id)
        event(run_dir,'training','STARTED',family=family,candidate=candidate,seed=seed,phase=phase,seconds_allowed=allowed)
        try:
            if family=='LightGBM':
                result=fit_lightgbm(dev,candidate,cfg,seed,directory,allowed)
            else:
                result=fit_tabm(dev,candidate,cfg,seed,directory,allowed,pp.cardinalities,bins)
            return result
        except Exception as e:
            failure={'family':family,'seed':seed,'candidate':candidate,'reason':str(e),'traceback':traceback.format_exc()}
            write_json(directory/'failure.json',failure)
            failures.append(failure)
            return None
        finally:
            budget.finish()
            result_path=directory/'result.json'
            if result_path.exists():
                recorded=read_json(result_path)
                recorded['cumulative_trial_seconds']=budget.state['trials'].get(trial_id,0)
                write_json(result_path,recorded)
    # Both first candidates before any remaining candidate.
    for i in range(3):
        for family,key in [('LightGBM','lightgbm'),('TabM','tabm')]:
            record=fit(family,cfg[key]['candidates'][i],cfg['seed_search'],'SEARCH')
            if record is not None:
                records[family].append(record)
    selected={}
    for family in records:
        try:
            selected[family]=[select_best(records[family])]
        except Blocked:
            continue
    write_json(run_dir/'search_results.json',{'records':records,'failures':failures})
    # Extra seeds in prescribed order. No ranking or selecting a seed.
    for seed in cfg['replication_seeds']:
        for family in ['LightGBM','TabM']:
            if family not in selected:
                continue
            key='lightgbm' if family=='LightGBM' else 'tabm'
            # Reserve no additional search; skip a replication if even its cap is unavailable.
            if budget.remaining(family)<cfg[key]['fit_seconds']:
                continue
            record=fit(family,selected[family][0]['config'],seed,'FIXED_CONFIGURATION_REPLICATION')
            if record and record['eligible']:
                selected[family].append(record)
    write_json(run_dir/'constant.json',{'p':float(np.mean(dev.y_A)),'fitted_partition':'A','label_hash':array_hash(dev.y_A)})
    write_json(run_dir/'selected_models.json',selected)
    del dev,bins
    gc.collect()
    return selected


def calibrate(data,pp,selected,cfg,run_dir):
    ids=data.splits['C']
    num,cat=pp.transform(*data.features(ids))
    y=data.labels('C')
    records={}
    for family,fits in selected.items():
        for model in fits:
            key=f'{family}_seed{model["seed"]}'
            p=predict(model,num,cat,cfg['cpu_threads'])
            cal=PositiveSigmoid().fit(p,y,cfg['calibration'])
            np.save(run_dir/'calibration'/f'{key}_raw.npy',p)
            info={'calibrator':cal.to_dict(), 'reliability_edges':{
                'raw':reliability_edges(p,cfg['evaluation']['reliability_bins']).tolist(),
                'calibrated':reliability_edges(cal.predict(p),cfg['evaluation']['reliability_bins']).tolist()},
                'calibration_index_hash':array_hash(ids),'base_checkpoint_sha256':model['checkpoint_sha256']}
            write_json(run_dir/'calibration'/f'{key}.json',info)
            records[key]=info
    constant=read_json(run_dir/'constant.json')['p']
    write_json(run_dir/'calibration'/'Constant.json',{'reliability_edges':reliability_edges(np.full(len(y),constant),cfg['evaluation']['reliability_bins']).tolist()})
    return records


def final_evaluate(data,pp,selected,calibrators,cfg,run_dir):
    manifest=run_dir/'FROZEN_MANIFEST.json'
    verify_freeze(manifest)
    complete=run_dir/'TEST_EVALUATION.json'
    if complete.exists():
        old=read_json(complete)
        if old['manifest_sha256']!=sha256(manifest):
            raise Blocked('Existing test evaluation has a different manifest.')
        for name,expected in old['outputs'].items():
            if sha256(run_dir/name)!=expected:
                raise Blocked(f'Cached test result changed: {name}')
        return old
    access=run_dir/'TEST_ACCESS_STARTED.json'
    if not access.exists():
        immutable_json(access,{'manifest_sha256':sha256(manifest),'utc':now(),'D_status':'NOW_SEEN_FOR_THIS_FROZEN_PROTOCOL'})
    elif read_json(access)['manifest_sha256']!=sha256(manifest):
        raise Blocked('D already accessed under another freeze.')
    ids=data.splits['D']
    dates=data.dates[ids]
    prediction_path=run_dir/'predictions_D.parquet'
    pending=prediction_path.with_suffix('.tmp.parquet')
    pre_receipt=run_dir/'PREDICTION_RECEIPT.json'
    if not prediction_path.exists() and pre_receipt.exists():
        r=read_json(pre_receipt)
        if not pending.exists() or r['manifest_sha256']!=sha256(manifest) or sha256(pending)!=r['sha256']:
            raise Blocked('Incomplete prediction commit cannot be verified; preserve seen-test state.')
        pending.replace(prediction_path)
    if not prediction_path.exists():
        num,cat=pp.transform(*data.features(ids))
        parts=[]
        for family,fits in selected.items():
            for fit in fits:
                key=f'{family}_seed{fit["seed"]}'
                p=predict(fit,num,cat,cfg['cpu_threads'])
                cal=PositiveSigmoid(**calibrators[key]['calibrator'])
                parts.append(pd.DataFrame({'row_id':ids,'time':dates,'model':family,'seed':fit['seed'],'p_raw':p,'p_cal':cal.predict(p)}))
        p=read_json(run_dir/'constant.json')['p']
        parts.append(pd.DataFrame({'row_id':ids,'time':dates,'model':'Constant','seed':42,'p_raw':p,'p_cal':p}))
        frame=pd.concat(parts,ignore_index=True)
        temp=prediction_path.with_suffix('.tmp.parquet')
        frame.to_parquet(temp,index=False)
        immutable_json(run_dir/'PREDICTION_RECEIPT.json',{'sha256':sha256(temp),'manifest_sha256':sha256(manifest),'row_count':len(frame),'labels_in_file':False})
        temp.replace(prediction_path)
    else:
        prediction_receipt=read_json(run_dir/'PREDICTION_RECEIPT.json')
        if sha256(prediction_path)!=prediction_receipt['sha256'] or prediction_receipt['manifest_sha256']!=sha256(manifest):
            raise Blocked('Frozen predictions changed.')
        frame=pd.read_parquet(prediction_path)
    # Coverage checks precede final label connection.
    expected_pairs={(family,model['seed']) for family,fits in selected.items() for model in fits}|{('Constant',42)}
    if set(frame.groupby(['model','seed']).groups)!=expected_pairs:
        raise Blocked('D prediction model/seed coverage mismatch')
    for _,part in frame.groupby(['model','seed'],sort=False):
        if not np.array_equal(part.row_id.to_numpy(),ids) or part.row_id.duplicated().any() or not np.array_equal(part.time.to_numpy(dtype='datetime64[D]'),dates):
            raise Blocked('D prediction row coverage/order mismatch')
        from .models import check_probabilities
        check_probabilities(part[['p_raw','p_cal']].to_numpy())
    y=data.final_labels(run_dir)
    rows,weeks,bins,ties=[],[],[],{}
    pairs={}
    for (family,seed),part in frame.groupby(['model','seed'],sort=False):
        key=f'{family}_seed{seed}'
        pair={'raw':part.p_raw.to_numpy(),'calibrated':part.p_cal.to_numpy()}
        variants=['raw'] if family=='Constant' else ['raw','calibrated']
        if family!='Constant':
            pairs[(family,int(seed))]=pair
            ties[key]=numeric_ties(pair['raw'],pair['calibrated'])
        for variant in variants:
            p=pair[variant]
            calibration_status='NOT_APPLICABLE' if family=='Constant' else calibrators[key]['calibrator']['status']
            tag={'model':family,'seed':int(seed),'variant':variant,'calibration_status':calibration_status,
                 'probability_source':'RAW_FALLBACK_CALIBRATION_FAILED' if variant=='calibrated' and calibration_status!='FITTED' else variant}
            rows.append({**tag,**metrics(y,p)})
            weeks.extend([{**tag,**r} for r in weekly(y,p,dates,cfg['evaluation']['small_week_n'])])
            edges=read_json(run_dir/'calibration'/'Constant.json')['reliability_edges'] if family=='Constant' else calibrators[key]['reliability_edges'][variant]
            bins.extend([{**tag,**r} for r in reliability(y,p,np.asarray(edges))])
    table=pd.DataFrame(rows)
    table.to_csv(run_dir/'metrics.csv',index=False)
    pd.DataFrame(weeks).to_csv(run_dir/'weekly_metrics.csv',index=False)
    pd.DataFrame(bins).to_csv(run_dir/'reliability_bins.csv',index=False)
    write_json(run_dir/'calibration_numerics.json',ties)
    shared=sorted(set(table.loc[table.model=='LightGBM','seed'])&set(table.loc[table.model=='TabM','seed']))
    matched=table[table.model.isin(['LightGBM','TabM'])&table.seed.isin(shared)]
    summary=matched.groupby(['model','variant'])[['roc_auc','average_precision','log_loss','brier']].agg(['mean','std','count'])
    summary.to_csv(run_dir/'matched_seed_summary.csv')
    write_json(run_dir/'matched_seeds.json',{'seeds':shared,'n':len(shared),'interpretation':'seed variability, not data-sampling uncertainty; no seed ensembling or selection'})
    if ('LightGBM',42) in pairs and ('TabM',42) in pairs:
        bootstrap,draws=moving_block_bootstrap(y,dates,pairs[('LightGBM',42)],pairs[('TabM',42)],cfg['evaluation'])
        write_json(run_dir/'bootstrap_summary.json',bootstrap)
        draws.to_csv(run_dir/'bootstrap_replicates.csv',index=False)
    portfolios,stress,rounding,controls=evaluate_portfolios(pairs,y,ids,dates,read_json(run_dir/'constant.json')['p'],cfg['portfolio'])
    for output in [portfolios,stress]:
        output['calibration_status']=[calibrators.get(f'{model}_seed{int(seed)}',{}).get('calibrator',{}).get('status','NOT_APPLICABLE') if pd.notna(seed) else 'NOT_APPLICABLE' for model,seed in zip(output.model,output.seed)]
    portfolios.to_csv(run_dir/'portfolio_metrics.csv',index=False)
    stress.to_csv(run_dir/'stress_scenarios.csv',index=False)
    rounding.to_csv(run_dir/'selection_rounding.csv',index=False)
    controls.to_csv(run_dir/'random_control_summary.csv',index=False)
    verify_freeze(manifest)
    outputs=[prediction_path,*run_dir.glob('*.csv'),run_dir/'calibration_numerics.json',run_dir/'matched_seeds.json']
    if (run_dir/'bootstrap_summary.json').exists():
        outputs.append(run_dir/'bootstrap_summary.json')
    receipt={'status':'COMPLETE','manifest_sha256':sha256(manifest),'utc':now(),
             'official_test_evaluation_records':1,'outputs':{p.name:sha256(p) for p in outputs}}
    immutable_json(complete,receipt)
    return receipt


def run(cfg):
    from .reporting import report
    from .absbox_demo import run_absbox
    run_dir=project_path(cfg['output_dir'])/cfg['run_id']
    run_dir.mkdir(parents=True,exist_ok=True)
    for sub in ['splits','calibration','fits']:
        (run_dir/sub).mkdir(exist_ok=True)
    started=time.monotonic()
    if (run_dir/'acceptance.json').exists() and (run_dir/'TEST_EVALUATION.json').exists():
        verify_freeze(run_dir/'FROZEN_MANIFEST.json')
        report(run_dir)
        return read_json(run_dir/'acceptance.json')
    immutable_json(run_dir/'config.json',cfg)
    if not (run_dir/'hardware.json').exists():
        write_json(run_dir/'hardware.json',hardware())
    acceptance={'core_status':'CORE_PARTIAL','extension_status':'EXTENSION_BLOCKED','missing':[],
                'run_dir':str(run_dir),'started_utc':now(),'test_accessed':(run_dir/'TEST_ACCESS_STARTED.json').exists()}
    try:
        event(run_dir,'data','ACQUIRING')
        folder=acquire(cfg,run_dir)
        data=Dataset(folder)
        audit=data.audit(run_dir)
        event(run_dir,'data','AUDITED',splits=audit['splits'])
        pp_path=run_dir/'preprocessor.joblib'
        if (run_dir/'FROZEN_MANIFEST.json').exists():
            verify_freeze(run_dir/'FROZEN_MANIFEST.json')
            pp=joblib.load(pp_path)
            selected=read_json(run_dir/'selected_models.json')
            calibrators={p.stem:read_json(p) for p in (run_dir/'calibration').glob('*_seed*.json')}
        else:
            if (run_dir/'TEST_ACCESS_STARTED.json').exists():
                raise Blocked('Test already accessed but freeze missing; cannot restart development.')
            # Freeze all choices before first formal fit.
            immutable_json(run_dir/'PROTOCOL_LOCK.json',{'config':cfg,'code':source_hashes(),'split_hash':sha256(run_dir/'split_manifest.json')})
            if pp_path.exists():
                pp=joblib.load(pp_path)
            else:
                pp=Preprocessor().fit(*data.features(data.splits['A']))
                joblib.dump(pp,pp_path)
            write_json(run_dir/'input_dimensions.json',pp.input_dimensions)
            if (run_dir/'selected_models.json').exists():
                selected=read_json(run_dir/'selected_models.json')
            else:
                selected=train_and_select(data,pp,cfg,run_dir)
            if not selected:
                raise Blocked('No model produced a valid B-selected checkpoint.')
            calibrators=calibrate(data,pp,selected,cfg,run_dir)
            files=[pp_path,run_dir/'config.json',run_dir/'PROTOCOL_LOCK.json',run_dir/'data_audit.json',run_dir/'feature_manifest.json',run_dir/'split_manifest.json',run_dir/'selected_models.json',run_dir/'constant.json',
                   *list((run_dir/'splits').glob('*.npy')),*list((run_dir/'calibration').glob('*'))]
            files += [Path(r['checkpoint']) for values in selected.values() for r in values]
            if (run_dir/'bins.joblib').exists():
                files.append(run_dir/'bins.joblib')
            freeze(run_dir/'FROZEN_MANIFEST.json',files,{'protocol_id':cfg['protocol_id'],'evaluation':cfg['evaluation'],'portfolio':cfg['portfolio'],'data_files':audit['files'],'D_status':'UNSEEN_BEFORE_THIS_FREEZE'})
        final_evaluate(data,pp,selected,calibrators,cfg,run_dir)
        acceptance['test_accessed']=True
        missing=[]
        for family in ['LightGBM','TabM']:
            if family not in selected or not any(x['seed']==42 for x in selected[family]):
                missing.append(f'{family} seed=42 missing')
        if any(r['calibrator']['status']!='FITTED' for r in calibrators.values()):
            missing.append('One or more calibrators failed; raw fallback explicitly retained')
        qa=ROOT/'logs'/'tests_receipt.json'
        if not qa.exists() or read_json(qa).get('exit_code')!=0 or read_json(qa).get('code')!=source_hashes():
            missing.append('Current-code automated test receipt absent')
        acceptance['missing']=missing
        acceptance['core_status']='CORE_COMPLETE' if not missing else 'CORE_PARTIAL'
    except Exception as error:
        acceptance['missing'].append(str(error))
        write_json(run_dir/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'utc':now()})
        if not (run_dir/'data_audit.json').exists():
            write_json(run_dir/'data_acquisition_status.json',{'status':'BLOCKED_DATA_UNAVAILABLE',**LIMITATIONS,
                      'actual_dimensions':None,'actual_dates':None,'split_sizes':None,'file_hashes':None,'D_labels_read':False})
        event(run_dir,'core','PARTIAL',reason=str(error))
    acceptance['test_accessed']=(run_dir/'TEST_ACCESS_STARTED.json').exists()
    acceptance['seconds_this_invocation']=time.monotonic()-started
    write_json(run_dir/'acceptance.json',acceptance)
    # First core report, then the bounded, independent synthetic extension.
    report(run_dir)
    extension=run_absbox(cfg['absbox'],run_dir/'absbox')
    acceptance['extension_status']=extension['status']
    acceptance['extension_reason']=extension.get('reason')
    acceptance['seconds_this_invocation']=time.monotonic()-started
    write_json(run_dir/'acceptance.json',acceptance)
    acceptance=refresh_acceptance_details(run_dir)
    report(run_dir)
    return acceptance


def refresh_acceptance_details(run_dir):
    """Reconcile machine status with cached artifacts; never execute a stage."""
    run_dir=Path(run_dir)
    state=read_json(run_dir/'acceptance.json')
    qa=ROOT/'logs'/'tests_receipt.json'
    tests=read_json(qa) if qa.exists() else {}
    selected=read_json(run_dir/'selected_models.json') if (run_dir/'selected_models.json').exists() else {}
    cals=[read_json(p)['calibrator'] for p in (run_dir/'calibration').glob('*_seed*.json')]
    requirements={
        'real_data_audit':(run_dir/'data_audit.json').exists(),
        'lightgbm_seed42':any(x.get('seed')==42 and x.get('eligible') for x in selected.get('LightGBM',[])),
        'tabm_seed42':any(x.get('seed')==42 and x.get('eligible') for x in selected.get('TabM',[])),
        'independent_calibration':len(cals)>=2 and all(x['status']=='FITTED' for x in cals),
        'frozen_final_test':(run_dir/'FROZEN_MANIFEST.json').exists() and (run_dir/'TEST_EVALUATION.json').exists(),
        'real_portfolio_analysis':(run_dir/'portfolio_metrics.csv').exists(),
        'report':(run_dir/'reports'/'PORTFOLIO_REPORT.md').exists(),
        'current_code_tests':tests.get('exit_code')==0 and tests.get('code')==source_hashes()}
    state['requirements']=requirements
    state['missing_requirements']=[k for k,v in requirements.items() if not v]
    state['core_status']='CORE_COMPLETE' if all(requirements.values()) else 'CORE_PARTIAL'
    state['test_accessed']=(run_dir/'TEST_ACCESS_STARTED.json').exists()
    state['formal_fit_counts']={family:len(list((run_dir/'fits').glob(f'{family}_*/result.json'))) for family in ['LightGBM','TabM']}
    state['formal_evaluated_seed_count']=read_json(run_dir/'matched_seeds.json')['n'] if (run_dir/'matched_seeds.json').exists() else 0
    state['tests_receipt']=str(qa)
    state['observed_data_dates_and_sizes_available']=(run_dir/'split_manifest.json').exists()
    write_json(run_dir/'acceptance.json',state)
    return state
