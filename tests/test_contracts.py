import copy
import inspect
import itertools
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
import torch

from credit_mvp.absbox_demo import annual_to_monthly, validate_endpoint
from credit_mvp.calibration import PositiveSigmoid, numeric_ties
from credit_mvp.common import Blocked, Budget, freeze, immutable_json, read_json, verify_freeze, write_json
from credit_mvp.data import Dataset, DevelopmentData, Preprocessor, split_four
from credit_mvp.evaluation import metrics, moving_block_bootstrap, reliability, reliability_edges, week_start
from credit_mvp.models import (fit_lightgbm, fit_tabm, head_loss, head_probabilities,
                               make_bins, make_tabm, predict_tabm_model, select_best)
from credit_mvp.pipeline import synthetic_development
from credit_mvp.portfolio import loss_metrics, random_mask, select_mask, stress_probability


@pytest.fixture
def config():
    from credit_mvp.common import ROOT, config_load
    return config_load(ROOT/'configs'/'mvp.yaml')


@pytest.fixture
def fixture_dataset(tmp_path):
    n=80
    rng=np.random.default_rng(123)
    dates=np.repeat(np.arange(18000,18020),4)
    np.save(tmp_path/'x_num.npy',rng.normal(size=(n,3)).astype(np.float32))
    np.save(tmp_path/'x_cat.npy',rng.integers(0,4,size=(n,2)))
    np.save(tmp_path/'x_meta.npy',np.c_[dates,np.arange(n)*3])
    np.save(tmp_path/'y.npy',np.tile([0,1],40))
    (tmp_path/'info.json').write_text(json.dumps({'name':'homecredit-default','task_type':'binclass'}))
    split=tmp_path/'splits'/'default'
    split.mkdir(parents=True)
    for name,ids in [('train',np.arange(0,32)),('val',np.arange(32,64)),('test',np.arange(64,80))]:
        np.save(split/f'{name}.npy',ids)
    return tmp_path


def test_four_splits_and_whitelist(fixture_dataset):
    data=Dataset(fixture_dataset)
    for a,b in itertools.combinations('ABCD',2):
        assert not np.intersect1d(data.splits[a],data.splits[b]).size
    for a,b in zip('ABC','BCD'):
        assert data.dates[data.splits[a]].max()<data.dates[data.splits[b]].min()
    assert np.array_equal(data.splits['D'],np.arange(64,80))
    assert not set(data.dates[data.splits['B']])&set(data.dates[data.splits['C']])
    num,cat=data.features(data.splits['A'])
    assert num.shape==(32,3) and cat.shape==(32,2)
    with pytest.raises(Blocked):
        data.labels('D')
    with pytest.raises(FileNotFoundError):
        data.final_labels(fixture_dataset/'unfrozen')


def test_bad_time_or_overlap_blocks():
    dates=np.arange(12)
    with pytest.raises(Blocked):
        split_four(dates,np.array([0,1,6]),np.array([2,3,4,5]),np.array([7,8]))
    with pytest.raises(Blocked):
        split_four(dates,np.array([0,1]),np.array([1,2,3,4]),np.array([8,9]))


def test_isolated_labels_do_not_reach_training_or_selection(fixture_dataset,config):
    from credit_mvp.data import development
    data=Dataset(fixture_dataset)
    pp=Preprocessor().fit(*data.features(data.splits['A']))
    first=development(data,pp)
    del data
    labels=np.load(fixture_dataset/'y.npy')
    labels[48:]=1-labels[48:]  # C and D only
    np.save(fixture_dataset/'y.npy',labels)
    second=development(Dataset(fixture_dataset),pp)
    assert set(first.__dataclass_fields__)=={'num_A','cat_A','y_A','num_B','cat_B','y_B'}
    for field in first.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(first,field),getattr(second,field))
    for trainer in [fit_lightgbm,fit_tabm]:
        parameters=inspect.signature(trainer).parameters
        assert 'y_D' not in parameters and 'test_callback' not in parameters and 'dataset' not in parameters
    records=[{'eligible':True,'config':{'id':'L2'},'B':{'roc_auc':.7,'log_loss':.4}},
             {'eligible':True,'config':{'id':'L1'},'B':{'roc_auc':.7,'log_loss':.4}}]
    assert select_best(records)['config']['id']=='L1'
    c_pred=np.linspace(.1,.9,16)
    c_y=np.tile([0,1],8)
    before=PositiveSigmoid().fit(c_pred,c_y,config['calibration']).to_dict()
    labels[64:]=np.random.default_rng(999).integers(0,2,16)
    np.save(fixture_dataset/'y.npy',labels)
    after=PositiveSigmoid().fit(c_pred,c_y,config['calibration']).to_dict()
    assert before==after


def test_preprocessing_train_only_missing_unknown_constant_roundtrip(tmp_path):
    a=np.array([[1,np.nan,7],[2,np.nan,7],[3,8,7]],dtype=np.float32)
    c=np.array([[1],[2],[np.nan]])
    pp=Preprocessor().fit(a,c)
    assert pp.keep.tolist()==[True,False,False]
    old=copy.deepcopy(pp.__dict__)
    x,cat=pp.transform(np.array([[np.inf,999,99],[4,np.nan,7]]),np.array([[999],[np.nan]]))
    assert np.isfinite(x).all() and cat[:,0].tolist()==[0,1]
    assert pp.fit_hash==old['fit_hash']
    joblib.dump(pp,tmp_path/'pp.joblib')
    np.testing.assert_array_equal(x,joblib.load(tmp_path/'pp.joblib').transform(np.array([[np.inf,999,99],[4,np.nan,7]]),np.array([[999],[np.nan]]))[0])


def test_per_head_loss_and_probability():
    logits=torch.tensor([[2.,-1.,0.],[-3.,.3,1.1]],requires_grad=True)
    y=torch.tensor([1.,0.])
    manual=sum(torch.nn.functional.binary_cross_entropy_with_logits(logits[:,i],y) for i in range(3))/3
    torch.testing.assert_close(head_loss(logits,y),manual)
    torch.testing.assert_close(head_probabilities(logits),torch.sigmoid(logits).mean(1))
    assert not torch.allclose(head_probabilities(logits),torch.sigmoid(logits.mean(1)))


def test_official_tabm_roundtrip_without_training(tmp_path,config):
    # Forward pass only; the single brief training smoke is executed separately.
    torch.set_num_threads(2)
    dev,pp=synthetic_development()
    bins=make_bins(dev.num_A,config['tabm']['n_bins'])
    model=make_tabm(bins,pp.cardinalities,config,.1,'cpu')
    p=predict_tabm_model(model,dev.num_B[:5],dev.cat_B[:5],5,'cpu')
    torch.save(model.state_dict(),tmp_path/'weights.pt')
    restored=make_tabm(bins,pp.cardinalities,config,.1,'cpu')
    restored.load_state_dict(torch.load(tmp_path/'weights.pt',weights_only=True))
    actual=predict_tabm_model(restored,dev.num_B[:5],dev.cat_B[:5],5,'cpu')
    np.testing.assert_allclose(p,actual,atol=1e-7)
    assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all()


def test_positive_sigmoid_failure_and_ties(config):
    p=np.linspace(.01,.99,100)
    y=np.random.default_rng(4).binomial(1,p)
    cal=PositiveSigmoid().fit(p,y,config['calibration'])
    assert cal.status=='FITTED' and cal.slope>0
    assert np.all(np.diff(cal.predict(p))>=0)
    failure=PositiveSigmoid().fit(p,np.zeros(100),config['calibration'])
    assert failure.status=='FAILED_SINGLE_CLASS'
    np.testing.assert_array_equal(failure.predict(p),p)
    extreme=np.array([0,1e-9,1e-7,.5,1-1e-7,1-1e-9,1])
    out=cal.predict(extreme)
    info=numeric_ties(extreme,out)
    assert info['nondecreasing'] and info['clipped_input_count']==4
    assert len(np.unique(out))<len(np.unique(extreme))
    np.testing.assert_array_equal(np.lexsort((np.arange(7),extreme,out)),np.arange(7))


def test_reliability_edges_do_not_use_test_labels():
    edges=reliability_edges(np.full(20,.3))
    assert edges[0]==0 and edges[-1]==1 and len(edges)<11
    p=np.array([0,.3,.8,1.])
    a=reliability(np.array([0,0,1,1]),p,edges)
    b=reliability(np.array([1,1,0,0]),p,edges)
    assert [r['n'] for r in a]==[r['n'] for r in b]
    assert sum(r['n'] for r in a)==4
    assert metrics(np.zeros(4),p)['roc_auc'] is None


@pytest.fixture
def pool_fixture():
    p=np.array([.1,.1,.8,.05,.2,.3,.7,.9])
    y=np.array([0,1,1,0,0,1,1,0])
    dates=np.array(['2020-06-01']*3+['2020-06-08']*5,dtype='datetime64[D]')
    return p,y,dates,np.arange(1,9)


@pytest.mark.parametrize('q,ids,el,loss,error',[(.5,[1,4,5],.14,0,.0466666666667),(.7,[1,2,4,5,6],.3,.8,-.1),(.9,[1,2,4,5,6,7],.58,1.2,-.1033333333333),(1.,list(range(1,9)),1.26,1.6,-.0425)])
def test_portfolio_independent_arithmetic_fixture(pool_fixture,q,ids,el,loss,error):
    p,y,dates,row=pool_fixture
    mask,audit=select_mask(p,row,dates,q)
    assert row[mask].tolist()==ids
    result=loss_metrics(p,y,mask)
    assert result['EL_hat']==pytest.approx(el)
    assert result['Loss_proxy']==pytest.approx(loss)
    assert result['error_per_exposure']==pytest.approx(error)
    assert result['exposure']==len(ids)
    assert sum(a['selected'] for a in audit)==len(ids)


def test_calibrated_same_pool_and_random_equal_exposure(pool_fixture):
    p,y,dates,row=pool_fixture
    mask,_=select_mask(p,row,dates,.7)
    calibrated=2*p/(1+p)
    mask_cal,_=select_mask(calibrated,row,dates,.7)
    np.testing.assert_array_equal(mask,mask_cal)
    result=loss_metrics(calibrated,y,mask)
    assert result['EL_hat']==pytest.approx(.5014985014985015)
    assert result['Loss_proxy']==.8
    assert result['error_per_exposure']==pytest.approx(-.0597002997002997)
    second=np.array([.4,.3,.2,.4,.3,.2,.1,.5])
    mask2,_=select_mask(second,row,dates,.7)
    assert row[mask2].tolist()==[2,3,5,6,7]
    assert mask2.sum()==mask.sum()==5
    for seed in range(1000,1020):
        random=random_mask(row,dates,.7,seed)
        assert random[:3].sum()==2 and random[3:].sum()==3
    for odds,lgd,expected in [(1,.4,.3),(1.5,.6,.6137499828),(2,.8,1.0029970030)]:
        assert np.sum(stress_probability(p,odds)[mask])*lgd==pytest.approx(expected)


def test_empty_selected_week_and_q1():
    p=np.array([.2,.8])
    dates=np.array(['2020-06-01','2020-06-08'],dtype='datetime64[D]')
    mask,audit=select_mask(p,np.arange(2),dates,.7)
    assert not mask.any() and all(x['zero_selected'] for x in audit)
    assert loss_metrics(p,np.array([0,1]),mask)['error_per_exposure'] is None
    full,_=select_mask(p,np.arange(2),dates,1)
    assert full.all()


def test_bootstrap_shared_blocks_and_gaps(config):
    dates=np.repeat(np.arange(np.datetime64('2020-06-01'),np.datetime64('2020-07-27'),np.timedelta64(7,'D')),4)
    y=np.tile([0,0,1,1],8)
    p=np.tile([.1,.3,.6,.8],8)
    cfg={**config['evaluation'],'bootstrap_replicates':20}
    same={'raw':p,'calibrated':p}
    status,draws=moving_block_bootstrap(y,dates,same,same,cfg)
    assert status['status']=='COMPLETE'
    assert np.all(draws[['auc_diff','log_loss_diff']].to_numpy()==0)
    assert (draws.groupby('replicate').blocks.nunique()==1).all()
    dates[-4:]+=np.timedelta64(7,'D')
    status,_=moving_block_bootstrap(y,dates,same,same,cfg)
    assert status['status']=='NOT_EXECUTED_GAPPED_CALENDAR'


def test_freeze_and_idempotence_detect_mutation(tmp_path):
    file=tmp_path/'checkpoint'
    file.write_bytes(b'fixed')
    freeze(tmp_path/'frozen.json',[file],{'evaluation':'fixed'})
    verify_freeze(tmp_path/'frozen.json')
    freeze(tmp_path/'frozen.json',[file],{'evaluation':'fixed'})
    file.write_bytes(b'changed')
    with pytest.raises(Blocked):
        verify_freeze(tmp_path/'frozen.json')
    immutable_json(tmp_path/'stage.json',{'done':True})
    with pytest.raises(Blocked):
        immutable_json(tmp_path/'stage.json',{'done':False})


def test_budget_survives_restart(tmp_path):
    b=Budget(tmp_path/'budget.json',{'TabM':10})
    b.begin('TabM',5)
    b.finish()
    used=b.used('TabM')
    assert Budget(tmp_path/'budget.json',{'TabM':10}).used('TabM')==used


def test_trial_budget_not_reset_on_resume(tmp_path,monkeypatch):
    import credit_mvp.common as common
    clock=[1000.0]
    monkeypatch.setattr(common.time,'time',lambda:clock[0])
    b=Budget(tmp_path/'budget.json',{'TabM':14400})
    assert b.begin('TabM',2700,'T1_seed42')==2700
    clock[0]+=2640
    # Simulate process death: a fresh Budget conservatively charges active time.
    resumed=Budget(tmp_path/'budget.json',{'TabM':14400})
    assert resumed.begin('TabM',2700,'T1_seed42')==60
    resumed.finish()


def test_auth_placeholder_does_not_block_actual_audit(fixture_dataset):
    directory=fixture_dataset/'run'
    directory.mkdir()
    write_json(directory/'data_acquisition_status.json',{'status':'BLOCKED_DATA_UNAVAILABLE'})
    audited=Dataset(fixture_dataset).audit(directory)
    assert audited['status']=='DATA_AUDITED'
    assert read_json(directory/'data_audit.json')['status']=='DATA_AUDITED'


def test_current_official_nested_task_schema(fixture_dataset):
    import shutil
    official=fixture_dataset/'homecredit-default'
    official.mkdir()
    for path in list(fixture_dataset.iterdir()):
        if path==official:
            continue
        if path.is_dir():
            shutil.copytree(path,official/path.name)
        else:
            shutil.copy2(path,official/path.name)
    write_json(official/'info.json',{'task':{'type':'binclass','score':'roc-auc'}})
    data=Dataset(official)
    assert data.n==Dataset(fixture_dataset).n
    write_json(official/'info.json',{'task':{'type':'regression'}})
    with pytest.raises(Blocked):
        Dataset(official)


def test_final_evaluation_transaction_and_report_data(fixture_dataset,config,monkeypatch):
    """Synthetic engine contract only: no extra model training and no real D access."""
    import credit_mvp.pipeline as pipeline
    data=Dataset(fixture_dataset)
    run=fixture_dataset/'run'
    (run/'calibration').mkdir(parents=True)
    audit=data.audit(run)
    pp=Preprocessor().fit(*data.features(data.splits['A']))
    write_json(run/'constant.json',{'p':.25})
    write_json(run/'calibration'/'Constant.json',{'reliability_edges':[0,.25,1]})
    records={family:[{'family':family,'seed':42}] for family in ['LightGBM','TabM']}
    cal=PositiveSigmoid(status='FITTED')
    calibrators={f'{family}_seed42':{'calibrator':cal.to_dict(),'reliability_edges':{'raw':[0,.5,1],'calibrated':[0,.5,1]}} for family in records}
    monkeypatch.setattr(pipeline,'predict',lambda record,num,cat,threads:expit(num[:,0]+(.1 if record['family']=='TabM' else 0)))
    freeze(run/'FROZEN_MANIFEST.json',[run/'constant.json'],{'data_files':audit['files']})
    before=copy.deepcopy(records)
    first=pipeline.final_evaluate(data,pp,records,calibrators,config,run)
    second=pipeline.final_evaluate(data,pp,records,calibrators,config,run)
    assert first==second and records==before
    assert first['official_test_evaluation_records']==1
    predictions=pd.read_parquet(run/'predictions_D.parquet')
    assert 'y' not in predictions and len(predictions)==3*len(data.splits['D'])
    table=pd.read_csv(run/'metrics.csv')
    assert len(table)==5
    portfolios=pd.read_csv(run/'portfolio_metrics.csv')
    for _,group in portfolios[portfolios.model.isin(['LightGBM','TabM'])].groupby('q'):
        assert group.n.nunique()==1
    # Input hashes are checked at final-label gate, not just on initial audit.
    with (fixture_dataset/'y.npy').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(Blocked):
        data.final_labels(run)


def test_absbox_units_and_endpoint_guard():
    assert (1-annual_to_monthly(.1))**12==pytest.approx(.9)
    assert annual_to_monthly(0)==0 and annual_to_monthly(1)==1
    validate_endpoint({'engine_url':'http://127.0.0.1:8081','synthetic_only':True})
    with pytest.raises(Blocked):
        validate_endpoint({'engine_url':'http://remote.example:8081','synthetic_only':True})
    with pytest.raises(Blocked):
        validate_endpoint({'engine_url':'http://127.0.0.1:8081','synthetic_only':False})


@pytest.mark.integration
def test_real_absbox_receipt_if_executed():
    from credit_mvp.common import ROOT
    directory=ROOT/'runs'/'homecredit_temporal_v1_20261014'/'absbox'
    if not (directory/'status.json').exists() or read_json(directory/'status.json')['status']!='EXTENSION_COMPLETE':
        pytest.skip('REAL ENGINE NOT EXECUTED: integration is not PASS')
    status=read_json(directory/'status.json')
    assert status['actual_engine_run'] and not status['mock']
    assert status['reconciliation_passed']
    assert all(x['passed'] for x in read_json(directory/'reconciliation.json').values())
