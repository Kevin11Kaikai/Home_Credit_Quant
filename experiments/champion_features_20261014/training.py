"""Frozen, single-seed A/B feature-transfer experiment. No C/D interface."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
from pathlib import Path
import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pyarrow.parquet as pq
from common import HERE, PROJECT, ORIGINAL, DERIVED, read, write, sha, now, protected_hashes

VARIANTS = ['tabred_missing', 'champion_lgb', 'champion_cat']
ALGORITHM_FILES = ['common.py', 'features.py', 'training.py']


def numeric_fit(values, n_a):
    a = np.asarray(values[:n_a], dtype=float)
    valid = np.isfinite(a)
    median = float(np.median(a[valid])) if valid.any() else 0.
    filled = np.where(valid, a, median)
    mean, scale = float(filled.mean()), float(filled.std())
    missing_p = float((~valid).mean())
    return dict(median=median, mean=mean, scale=scale or 1., keep_numeric=bool(np.ptp(filled)>0),
                missing_p=missing_p, add_missing=bool(0<missing_p<1))


def numeric_transform(values, state):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    out = []
    if state['keep_numeric']:
        out.append(((np.where(valid, values, state['median'])-state['mean'])/state['scale']).astype(np.float32))
    if state['add_missing']:
        p = state['missing_p']
        out.append((((~valid)-p)/np.sqrt(p*(1-p))).astype(np.float32))
    return out


def category_fit(values, n_a):
    return {str(v): i+2 for i, v in enumerate(sorted({str(v) for v in values[:n_a] if v is not None}))}


def category_transform(values, vocab):
    return np.array([1 if v is None else vocab.get(str(v),0) for v in values], dtype=np.int64)


def variant_dir(variant):
    return DERIVED / variant / 'model_input'


def completed_input(folder):
    if not (folder/'state.json').exists():
        return False
    state = read(folder/'state.json')
    for name, expected in state['files'].items():
        assert sha(folder/name) == expected, f'Input changed: {folder/name}'
    return True


def build_missing():
    import joblib
    folder = variant_dir('tabred_missing')
    folder.mkdir(parents=True, exist_ok=True)
    if completed_input(folder):
        return
    pre = joblib.load(ORIGINAL/'preprocessor.joblib')
    a, b = [np.load(ORIGINAL/f'splits/{s}.npy') for s in ['A','B']]
    rows = np.r_[a,b]
    num, binary, cat = [np.load(PROJECT/f'data/homecredit-default/x_{n}.npy',mmap_mode='r') for n in ['num','bin','cat']]
    counts = np.zeros(num.shape[1]+binary.shape[1],dtype=np.int64)
    for start in range(0,len(a),8192):
        chunk = a[start:start+8192]
        counts += (~np.isfinite(np.column_stack([num[chunk],binary[chunk]]))).sum(axis=0)
    # Frozen original information retained, with one flag for every A-variable missing mask.
    flag_indices = np.flatnonzero((counts>0)&(counts<len(a)))
    probabilities = counts[flag_indices]/len(a)
    scales = np.sqrt(probabilities*(1-probabilities))
    n_base = int(pre.keep.sum())
    xn = np.lib.format.open_memmap(folder/'num.npy',mode='w+',dtype=np.float32,shape=(len(rows),n_base+len(flag_indices)))
    xc = np.lib.format.open_memmap(folder/'cat.npy',mode='w+',dtype=np.int64,shape=(len(rows),cat.shape[1]))
    for start in range(0,len(rows),8192):
        chunk = rows[start:start+8192]
        raw = np.column_stack([num[chunk],binary[chunk]])
        n,c = pre.transform(raw,cat[chunk])
        xn[start:start+len(chunk),:n_base] = n
        xn[start:start+len(chunk),n_base:] = ((~np.isfinite(raw[:,flag_indices]))-probabilities)/scales
        xc[start:start+len(chunk)] = c
    xn.flush();xc.flush()
    write(folder/'state.json',dict(variant='tabred_missing',n_A=len(a),n_B=len(b),base_numeric_count=n_base,
          numeric_count=xn.shape[1],categorical_count=xc.shape[1],cardinalities=pre.cardinalities,
          numeric_names=[f'tabred__num_{j}' for j in np.flatnonzero(pre.keep)]+[f'tabred__num_{j}__missing' for j in flag_indices],
          categorical_names=[f'tabred__cat_{j}' for j in range(cat.shape[1])],
          flag_source_columns=flag_indices.tolist(),flag_mean=probabilities.tolist(),flag_scale=scales.tolist(),
          preprocessing_reference_sha256=sha(ORIGINAL/'preprocessor.joblib'),fit_rows='A only',
          files={p:sha(folder/p) for p in ['num.npy','cat.npy']}))
    print('INPUT_READY tabred_missing',xn.shape,xc.shape,flush=True)


def build_champion(variant):
    folder = variant_dir('champion_'+variant)
    folder.mkdir(parents=True,exist_ok=True)
    if completed_input(folder):
        return
    schema = read(DERIVED/variant/'schema.json')
    n_a,n_b = schema['n_A'],schema['n_B']
    states, vocabs, numeric_names = {}, {}, []
    cat_names = schema['categorical']
    for item in schema['feature_tables']:
        path = HERE/item['path']
        assert sha(path)==item['sha256']
        table = pq.read_table(path)
        for name in table.column_names:
            if name in cat_names:
                vocabs[name] = category_fit(table[name].to_pylist(),n_a)
            else:
                state = numeric_fit(table[name].to_numpy(zero_copy_only=False),n_a)
                states[name] = state
                if state['keep_numeric']:numeric_names.append(name)
                if state['add_missing']:numeric_names.append(name+'__missing')
        del table
        gc.collect()
    xn=np.lib.format.open_memmap(folder/'num.npy',mode='w+',dtype=np.float32,shape=(n_a+n_b,len(numeric_names)))
    xc=np.lib.format.open_memmap(folder/'cat.npy',mode='w+',dtype=np.int64,shape=(n_a+n_b,len(cat_names)))
    ni={n:i for i,n in enumerate(numeric_names)}
    ci={n:i for i,n in enumerate(cat_names)}
    for item in schema['feature_tables']:
        table=pq.read_table(HERE/item['path'])
        for name in table.column_names:
            if name in vocabs:
                xc[:,ci[name]]=category_transform(table[name].to_pylist(),vocabs[name])
            else:
                vectors=numeric_transform(table[name].to_numpy(zero_copy_only=False),states[name])
                names=([name] if states[name]['keep_numeric'] else [])+([name+'__missing'] if states[name]['add_missing'] else [])
                for n,v in zip(names,vectors):xn[:,ni[n]]=v
        del table
        gc.collect()
    xn.flush();xc.flush()
    assert np.isfinite(xn).all()
    write(folder/'state.json',dict(variant='champion_'+variant,n_A=n_a,n_B=n_b,numeric_count=len(numeric_names),
          categorical_count=len(cat_names),numeric_names=numeric_names,categorical_names=cat_names,
          cardinalities=[len(vocabs[n])+2 for n in cat_names],numeric_states=states,vocabs=vocabs,
          fit_rows='A only',source_schema_sha256=sha(DERIVED/variant/'schema.json'),
          files={p:sha(folder/p) for p in ['num.npy','cat.npy']}))
    print('INPUT_READY champion_'+variant,xn.shape,xc.shape,flush=True)


def prepare():
    assert read(HERE/'feature_extraction_receipt.json')['status']=='AUTHOR_AGGREGATIONS_COMPLETE_AB'
    before=protected_hashes()
    build_missing()
    for v in ['lgb','cat']:build_champion(v)
    assert before==protected_hashes()


def protocol():
    path=HERE/'PROTOCOL_LOCK.json'
    code={p:sha(HERE/p) for p in ALGORITHM_FILES}
    inputs={str(q.relative_to(HERE)):sha(q) for q in
            [DERIVED/'mapping_AB.parquet',DERIVED/'y_A.npy',DERIVED/'y_B.npy',
             HERE/'SOURCE_MANIFEST.json',HERE/'download_receipt.json',HERE/'feature_extraction_receipt.json']}
    for v in VARIANTS:
        assert completed_input(variant_dir(v)),f'Missing prepared input: {v}'
    if path.exists():
        p=read(path)
        assert p['code']==code,'Experiment code changed after protocol lock'
        assert p['data_and_provenance_hashes']==inputs,'Data or provenance changed after protocol lock'
        for v in VARIANTS:
            assert p['input_state_hashes'][v]==sha(variant_dir(v)/'state.json')
        assert p['original_protected_hashes']==protected_hashes()
        return p
    selected=read(ORIGINAL/'selected_models.json')
    cfg=read(ORIGINAL/'config.json')
    p=dict(created_utc=now(),status='FROZEN_POSTHOC_AB_DEVELOPMENT_ONLY',seed=42,
           authorization='2026-10-14 user requested reproduction of Kaggle champion feature extraction and transfer to current neural model; separate follow-up, original evidence/budget unchanged',
           receiver='Official TabM; recommended default after optional clarification. No additional Transformer architecture assumed.',
           variants=VARIANTS,order=[f'{v}/{family}' for v in VARIANTS for family in ['LightGBM','TabM']],
           new_fits_per_family=3,hyperparameter_search=False,additional_seeds=False,
           models={k:next(r for r in selected[k] if r['seed']==42)['config'] for k in ['LightGBM','TabM']},
           model_configuration=cfg,selection='Existing model configurations fixed; checkpoints selected by B AUC as original protocol; report every feature variant, no winner-only reporting',
           metrics=['roc_auc','average_precision','log_loss','brier','mean_probability','observed_event_rate'],
           evaluation='B development metrics, already reused for selection; C/D never loaded; no new independent-test claim',
           effects=['missing vs original: explicit missingness','champion vs missing: entire representation package, including aggregates/schema/encoding, not one-column causal effect'],
           limits={'LightGBM_total_seconds':1800,'TabM_total_seconds':5400,'LightGBM_fit_seconds':1200,'TabM_fit_seconds':2700,'wall_seconds':7200,'cpu_threads':4},
           code=code,data_and_provenance_hashes=inputs,input_state_hashes={v:sha(variant_dir(v)/'state.json') for v in VARIANTS},
           original_protected_hashes=protected_hashes(),data_mapping_sha256=sha(DERIVED/'mapping_AB.parquet'))
    write(path,p)
    return p


def load_dev(variant):
    from credit_mvp.data import DevelopmentData
    folder=variant_dir(variant)
    state=read(folder/'state.json')
    n_a=state['n_A']
    x=np.load(folder/'num.npy',mmap_mode='r')
    c=np.load(folder/'cat.npy',mmap_mode='r')
    return DevelopmentData(x[:n_a],c[:n_a],np.load(DERIVED/'y_A.npy'),x[n_a:],c[n_a:],np.load(DERIVED/'y_B.npy')),state


def metrics(y,p):
    from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return dict(roc_auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),
                log_loss=float(log_loss(y,p,labels=[0,1])),brier=float(brier_score_loss(y,p)),
                mean_probability=float(p.mean()),observed_event_rate=float(y.mean()),n=len(y))


def replay_original():
    from credit_mvp.models import predict
    selected=read(ORIGINAL/'selected_models.json')
    dev,state=load_dev('tabred_missing')
    folder=HERE/'baseline_replay'
    folder.mkdir(exist_ok=True)
    for family in ['LightGBM','TabM']:
        target=folder/(family+'.json')
        if target.exists():
            saved=read(target)
            assert sha(folder/(family+'.npy'))==saved['prediction_sha256']
            continue
        record=next(r for r in selected[family] if r['seed']==42)
        p=predict(record,dev.num_B[:,:state['base_numeric_count']],dev.cat_B)
        m=metrics(dev.y_B,p)
        assert abs(m['roc_auc']-record['B']['roc_auc'])<1e-6
        assert abs(m['log_loss']-record['B']['log_loss'])<1e-6
        np.save(folder/(family+'.npy'),p)
        write(target,dict(variant='original',family=family,seed=42,status='REUSED_CHECKPOINT_B_REPLAY',
              metrics=m,checkpoint_sha256=record['checkpoint_sha256'],prediction_sha256=sha(folder/(family+'.npy')),
              new_training_seconds=0,checkpoint_selected_on_B=True,original_input_numeric=state['base_numeric_count'],
              original_input_categorical=state['categorical_count']))


def fit(variant,family,seconds):
    import joblib
    from credit_mvp.models import fit_lightgbm,fit_tabm,make_bins,predict
    p=protocol()
    cfg=p['model_configuration']
    dev,state=load_dev(variant)
    folder=HERE/'fits'/variant/family
    folder.mkdir(parents=True,exist_ok=True)
    if family=='LightGBM':
        record=fit_lightgbm(dev,p['models'][family],cfg,42,folder,seconds)
    else:
        bp=variant_dir(variant)/'bins.joblib'
        if bp.exists():bins=joblib.load(bp)
        else:
            bins=make_bins(dev.num_A,cfg['tabm']['n_bins'])
            joblib.dump(bins,bp)
        record=fit_tabm(dev,p['models'][family],cfg,42,folder,seconds,state['cardinalities'],bins)
    if not record.get('eligible'):
        write(folder/'evaluation.json',dict(status='NO_COMPLETED_B_CHECKPOINT',variant=variant,family=family))
        return
    prediction=predict(record,dev.num_B,dev.cat_B)
    np.save(folder/'predictions_B.npy',prediction)
    write(folder/'evaluation.json',dict(variant=variant,family=family,seed=42,status=record['status'],
          metrics=metrics(dev.y_B,prediction),prediction_sha256=sha(folder/'predictions_B.npy'),
          checkpoint_sha256=record['checkpoint_sha256'],new_training_seconds=record['seconds'],
          numeric_inputs=state['numeric_count'],categorical_inputs=state['categorical_count'],
          peak_rss_bytes=record['peak_rss_bytes'],gpu_peak_bytes=record['gpu_peak_bytes'],
          completed_epochs=record.get('epochs_completed'),best_epoch=record.get('best_epoch'),
          best_iteration=record.get('best_iteration'),checkpoint_selected_on_B=True))


def run():
    p=protocol()
    ledger_path=HERE/'budget.json'
    ledger=read(ledger_path) if ledger_path.exists() else dict(started_utc=now(),LightGBM=0.,TabM=0.,trials={})
    for key,trial in ledger['trials'].items():
        if trial['status']=='STARTED':
            # Unknown process outcome is charged conservatively, never reset.
            charge=trial['allocated_seconds']+300
            ledger[key.split('/')[1]]+=charge
            trial.update(status='INTERRUPTED_UNCONFIRMED',worker_seconds=charge,conservative_charge=True)
    write(ledger_path,ledger)
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'baseline'],check=True,cwd=PROJECT)
    for variant in VARIANTS:
        for family in ['LightGBM','TabM']:
            key=variant+'/'+family
            folder=HERE/'fits'/variant/family
            if (folder/'evaluation.json').exists():
                print('CACHED',key,flush=True)
                continue
            if key in ledger['trials']:
                print('PRESERVED_PREVIOUS_ATTEMPT',key,flush=True)
                continue
            elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(ledger['started_utc'])).total_seconds()
            seconds=min(p['limits'][family+'_fit_seconds'],p['limits'][family+'_total_seconds']-ledger[family],p['limits']['wall_seconds']-elapsed)
            if seconds<60:
                print('BUDGET_STOP',key,flush=True)
                continue
            folder.mkdir(parents=True,exist_ok=True)
            ledger['trials'][key]=dict(status='STARTED',started_utc=now(),allocated_seconds=seconds)
            write(ledger_path,ledger)
            start=time.monotonic()
            command=[sys.executable,str(Path(__file__).resolve()),'fit','--variant',variant,'--family',family,'--seconds',str(seconds)]
            print('START',key,flush=True)
            with (folder/'console.log').open('w',encoding='utf-8') as log:
                try:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=PROJECT,timeout=seconds+300)
                    code=result.returncode
                except subprocess.TimeoutExpired:
                    code=-999
            wall=time.monotonic()-start
            ledger[family]+=wall
            ledger['trials'][key].update(status='FINISHED' if code==0 else 'FAILED',exit_code=code,worker_seconds=wall)
            write(ledger_path,ledger)
            print('END',key,'exit',code,'seconds',round(wall,1),flush=True)
    summarize()


def summarize():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=protocol()
    records=[read(f) for f in sorted((HERE/'baseline_replay').glob('*.json'))]
    records += [read(f) for f in sorted((HERE/'fits').glob('*/*/evaluation.json')) if 'metrics' in read(f)]
    rows=[dict(variant=r['variant'],model=r['family'],seed=42,status=r['status'],
               new_training_seconds=r['new_training_seconds'],**r['metrics']) for r in records]
    frame=pd.DataFrame(rows)
    frame.to_csv(HERE/'development_metrics_B.csv',index=False)
    base=frame[frame.variant=='original'].set_index('model')
    for m in ['roc_auc','average_precision','log_loss','brier']:
        frame['delta_'+m+'_vs_original']=[getattr(row,m)-base.loc[row.model,m] for row in frame.itertuples()]
    frame.to_csv(HERE/'feature_effects_B.csv',index=False)
    dates=pq.read_table(DERIVED/'mapping_AB.parquet',columns=['date_decision','split']).to_pandas()
    weeks=pd.to_datetime(dates.loc[dates.split=='B','date_decision']).dt.to_period('W-SUN').astype(str).to_numpy()
    y=np.load(DERIVED/'y_B.npy')
    weekly=[]
    for r in records:
        f=(HERE/'baseline_replay'/(r['family']+'.npy') if r['variant']=='original' else HERE/'fits'/r['variant']/r['family']/'predictions_B.npy')
        assert sha(f)==r['prediction_sha256']
        probs=np.load(f)
        for w in np.unique(weeks):
            mask=weeks==w
            if len(np.unique(y[mask]))<2:
                weekly.append(dict(variant=r['variant'],model=r['family'],week=w,n=int(mask.sum()),roc_auc=None))
            else:
                weekly.append(dict(variant=r['variant'],model=r['family'],week=w,**metrics(y[mask],probs[mask])))
    pd.DataFrame(weekly).to_csv(HERE/'weekly_metrics_B.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric,title in zip(axes,['roc_auc','log_loss'],['B ROC-AUC (higher better)','B log loss (lower better)']):
        pivot=frame.pivot(index='variant',columns='model',values=metric).reindex(['original']+VARIANTS)
        pivot.plot.bar(ax=ax,color=['#477db3','#d29037'],rot=25)
        ax.set_title(title);ax.set_xlabel('');ax.grid(axis='y',alpha=.2)
        vals=pivot.to_numpy();lo=np.nanmin(vals);hi=np.nanmax(vals)
        ax.set_ylim(lo-(hi-lo)*.2-.001,hi+(hi-lo)*.25+.001)
    fig.suptitle('Feature transfer: reused development B, seed 42; not independent test evidence')
    fig.tight_layout();fig.savefig(HERE/'development_comparison.png',dpi=160);plt.close(fig)
    files={str(f.relative_to(HERE)):sha(f) for f in (HERE/'fits').glob('*/*/result.json')}
    files.update({str(f.relative_to(HERE)):sha(f) for f in (HERE/'fits').glob('*/*/predictions_B.npy')})
    files.update({n:sha(HERE/n) for n in ['development_metrics_B.csv','feature_effects_B.csv','weekly_metrics_B.csv','PROTOCOL_LOCK.json']})
    assert p['original_protected_hashes']==protected_hashes()
    expected=len(VARIANTS)*2+2
    partial=[r['variant']+'/'+r['family'] for r in records if r['status'] not in
             ['REUSED_CHECKPOINT_B_REPLAY','EARLY_STOPPED','MAX_EPOCHS','MAX_ROUNDS']]
    write(HERE/'acceptance.json',dict(status='MATCHED_TRANSFER_COMPLETE' if len(records)==expected and not partial else 'MATCHED_TRANSFER_PARTIAL',
          feature_extraction='AUTHOR_AGGREGATIONS_REPRODUCED_WITH_DOCUMENTED_AB_ADAPTATION',
          full_champion_model='NOT_REPRODUCED_PUBLIC_RELEASE_DEPENDS_ON_ADDITIONAL_FITTED_MODELS_AND_FINAL_FEATURE_MASKS',
          completed_metric_rows=len(records),expected_metric_rows=expected,seed_count=1,partial_training=partial,
          original_artifacts_unchanged=True,new_D_evaluations=0,C_used=False,
          interpretation='Posthoc development comparison. B selects checkpoints; no independent generalization or causal attribution to a single feature.',
          completed_utc=now(),files=files))
    print(frame[['variant','model','roc_auc','average_precision','log_loss','brier']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run','fit','baseline','report'])
    parser.add_argument('--variant',choices=VARIANTS)
    parser.add_argument('--family',choices=['LightGBM','TabM'])
    parser.add_argument('--seconds',type=float,default=1200)
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='run':run()
    elif args.stage=='fit':fit(args.variant,args.family,args.seconds)
    elif args.stage=='baseline':replay_original()
    else:summarize()
