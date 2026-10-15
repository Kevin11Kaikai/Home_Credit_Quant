from __future__ import annotations

import gc
import math
from pathlib import Path
import random
import time

import lightgbm as lgb
import numpy as np
import psutil
import rtdl_num_embeddings as rtdl
from sklearn.metrics import log_loss, roc_auc_score
import torch
from torch.nn import functional as F
from tabm import TabM

from .common import Blocked, read_json, sha256, write_json
from .data import DevelopmentData


def set_seed(seed,threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def head_loss(logits,y):
    if logits.ndim == 3:
        logits = logits.squeeze(-1)
    if logits.ndim != 2 or y.ndim != 1:
        raise ValueError('Expected [batch,heads] binary logits and [batch] labels')
    return F.binary_cross_entropy_with_logits(logits,y[:,None].expand_as(logits))


def head_probabilities(logits):
    if logits.ndim == 3:
        logits = logits.squeeze(-1)
    return torch.sigmoid(logits.float()).mean(dim=1)


def check_probabilities(p):
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
        raise FloatingPointError('Invalid predicted probabilities')


def validation_scores(y,p):
    check_probabilities(p)
    if len(np.unique(y))!=2:
        raise Blocked('AUC selection requires two classes in B.')
    return {'roc_auc':float(roc_auc_score(y,p)), 'log_loss':float(log_loss(y,p,labels=[0,1]))}


def selection_key(record):
    return (-record['B']['roc_auc'],record['B']['log_loss'],record['config']['id'])


def select_best(records):
    eligible = [r for r in records if r.get('eligible',False)]
    if not eligible:
        raise Blocked('No eligible completed validation checkpoint')
    return sorted(eligible,key=selection_key)[0]


def lgb_matrix(num,cat):
    return np.concatenate([num,cat.astype(np.float32)],axis=1)


def fit_lightgbm(dev: DevelopmentData, candidate, cfg, seed, directory, seconds, smoke=False):
    """Only A/B data accepted. No test labels or callbacks in this API."""
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():
        return read_json(directory/'result.json')
    set_seed(seed,cfg['cpu_threads'])
    start = time.monotonic()
    params = {**cfg['lightgbm']['common'],**{k:v for k,v in candidate.items() if k!='id'},
              'seed':seed,'bagging_seed':seed,'feature_fraction_seed':seed,'data_random_seed':seed,
              'num_threads':cfg['cpu_threads'],'metric':['auc','binary_logloss']}
    xa,xb = lgb_matrix(dev.num_A,dev.cat_A),lgb_matrix(dev.num_B,dev.cat_B)
    categories = list(range(dev.num_A.shape[1],xa.shape[1]))
    train = lgb.Dataset(xa,label=dev.y_A,categorical_feature=categories,free_raw_data=True)
    val = lgb.Dataset(xb,label=dev.y_B,categorical_feature=categories,reference=train,free_raw_data=True)
    history, stop_reason = {}, {'value':'MAX_ROUNDS'}
    peak_rss = psutil.Process().memory_info().rss
    deadline = start + seconds
    def budget_callback(env):
        nonlocal peak_rss
        peak_rss = max(peak_rss,psutil.Process().memory_info().rss)
        if time.monotonic() >= deadline:
            stop_reason['value'] = 'TIME_LIMIT'
            # Persist latest state even if stopped between early-stopping checks.
            env.model.save_model(str(directory/'interrupted_model.txt'))
            raise lgb.callback.EarlyStopException(env.iteration,env.evaluation_result_list)
    budget_callback.order = 25
    rounds = min(5,cfg['lightgbm']['max_rounds']) if smoke else cfg['lightgbm']['max_rounds']
    snapshot={'model':None}
    def preserve_native(env):
        snapshot['model']=env.model
        if (env.iteration+1)%25==0:
            env.model.save_model(str(directory/'latest_native.txt'))
            write_json(directory/'native_recovery.json',{'completed_iteration':env.iteration+1,'history':history,'params':params,
                       'resume_policy':'Preserve interrupted best checkpoint; native internal RNG cannot be serialized by public API. Do not silently refit this trial.'})
    preserve_native.order=24
    # A previous process kill with only a native snapshot is an explicit partial fit.
    if (directory/'latest_native.txt').exists():
        state=read_json(directory/'native_recovery.json')
        history=state['history']
        model=lgb.Booster(model_file=str(directory/'latest_native.txt'))
        stop_reason['value']='INTERRUPTED_NATIVE_CHECKPOINT_RETAINED'
    else:
        try:
            model = lgb.train(params,train,num_boost_round=rounds,valid_sets=[val],valid_names=['B'],
                              callbacks=[lgb.record_evaluation(history),preserve_native,lgb.early_stopping(cfg['lightgbm']['patience'],first_metric_only=True,verbose=False),budget_callback])
        except KeyboardInterrupt:
            model=snapshot['model']
            if model is None or not history.get('B',{}).get('auc'):
                raise
            model.save_model(str(directory/'latest_native.txt'))
            write_json(directory/'native_recovery.json',{'completed_iteration':len(history['B']['auc']),'history':history,'params':params,
                       'resume_policy':'Interrupted checkpoint retained; no silent full refit.'})
            stop_reason['value']='INTERRUPTED_NATIVE_CHECKPOINT_RETAINED'
    # Explicitly recover best B AUC among the completed rounds, with earliest tie.
    best_iteration = int(np.argmax(history['B']['auc']))+1
    if stop_reason['value']=='MAX_ROUNDS' and len(history['B']['auc'])<rounds:
        stop_reason['value']='EARLY_STOPPED'
    model.save_model(str(directory/'model.txt'),num_iteration=best_iteration)
    loaded = lgb.Booster(model_file=str(directory/'model.txt'))
    p = loaded.predict(xb,num_threads=cfg['cpu_threads'])
    assert np.allclose(p,model.predict(xb,num_iteration=best_iteration,num_threads=cfg['cpu_threads']),rtol=1e-12,atol=1e-12)
    result = {'family':'LightGBM','seed':seed,'config':candidate,'B':validation_scores(dev.y_B,p),
              'status':stop_reason['value'],'eligible':True,'converged':stop_reason['value']=='EARLY_STOPPED',
              'seconds':time.monotonic()-start,'best_iteration':best_iteration,'iterations_completed':len(history['B']['auc']),
              'checkpoint':str((directory/'model.txt').resolve()),'checkpoint_sha256':sha256(directory/'model.txt'),
              'peak_rss_bytes':peak_rss,'gpu_peak_bytes':0,'params':params,
              'optimizer':'LightGBM native gradient boosting; optimizer checkpoint not applicable'}
    write_json(directory/'curves.json',history)
    write_json(directory/'result.json',result)
    return result


def make_bins(num,n_bins):
    # A-only exact per-column quantiles, chunked for bounded RAM.
    bins=[]
    for start in range(0,num.shape[1],16):
        bins.extend(rtdl.compute_bins(torch.from_numpy(num[:,start:start+16].copy()),n_bins=n_bins))
    return bins


def make_tabm(bins,cardinalities,cfg,dropout,device):
    t=cfg['tabm']
    return TabM.make(n_num_features=len(bins),cat_cardinalities=cardinalities,
                     d_out=1,num_embeddings=rtdl.PiecewiseLinearEmbeddings(bins,d_embedding=t['d_embedding'],activation=False,version='B'),
                     n_blocks=t['n_blocks'],d_block=t['d_block'],k=t['k'],dropout=dropout,arch_type='tabm').to(device)


@torch.inference_mode()
def predict_tabm_model(model,num,cat,batch_size,device):
    model.eval()
    outputs=[]
    for start in range(0,len(num),batch_size):
        n=torch.from_numpy(num[start:start+batch_size]).to(device)
        c=torch.from_numpy(cat[start:start+batch_size]).to(device) if cat.shape[1] else None
        outputs.append(head_probabilities(model(n,c)).cpu().numpy())
    p=np.concatenate(outputs).astype(np.float64)
    check_probabilities(p)
    return p


def save_torch(path,payload):
    path=Path(path)
    tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp)
    tmp.replace(path)


def fit_tabm(dev: DevelopmentData,candidate,cfg,seed,directory,seconds,cardinalities,bins,smoke=False):
    """Sequential GPU fit, A/B only; FP32 and no torch.compile."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():
        return read_json(directory/'result.json')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cpu' and not smoke:
        raise Blocked('CUDA unavailable: TabM formal training not authorized on CPU; smoke only.')
    set_seed(seed,cfg['cpu_threads'])
    if device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    t=cfg['tabm']
    model=make_tabm(bins,cardinalities,cfg,candidate['dropout'],device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=candidate['learning_rate'],weight_decay=t['weight_decay'])
    batch=t['batch_size']
    history=[]
    epoch=0
    position=0
    permutation=None
    stale=0
    best_auc=-math.inf
    best_epoch=None
    resumed=False
    latest=directory/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location=device,weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch,position=state['epoch'],state['position']
        permutation=state['permutation'].cpu() if state['permutation'] is not None else None
        history=state['history']
        best_auc,best_epoch,stale=state['best_auc'],state['best_epoch'],state['stale']
        batch=state['batch']
        torch.set_rng_state(state['torch_rng'].cpu())
        if device=='cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        resumed=True
    peak_rss=psutil.Process().memory_info().rss
    status='MAX_EPOCHS'
    max_epochs=1 if smoke else t['max_epochs']
    def checkpoint():
        return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'position':position,
                'permutation':permutation,'history':history,'best_auc':best_auc,'best_epoch':best_epoch,'stale':stale,
                'batch':batch,'bins':bins,'cardinalities':cardinalities,'config':cfg,'candidate':candidate,'seed':seed,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],
                'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
    try:
        while epoch<max_epochs:
            if permutation is None:
                permutation=torch.randperm(len(dev.y_A))
                position=0
            losses=[]
            while position<len(permutation):
                if time.monotonic()-start>=seconds:
                    status='TIME_LIMIT'
                    save_torch(latest,checkpoint())
                    break
                effective=min(t['effective_batch_size'],len(permutation)-position)
                optimizer.zero_grad(set_to_none=True)
                try:
                    for offset in range(0,effective,batch):
                        idx=permutation[position+offset:position+min(offset+batch,effective)].numpy()
                        n=torch.from_numpy(dev.num_A[idx]).to(device)
                        c=torch.from_numpy(dev.cat_A[idx]).to(device) if dev.cat_A.shape[1] else None
                        y=torch.from_numpy(dev.y_A[idx].astype(np.float32)).to(device)
                        loss=head_loss(model(n,c),y)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite TabM training loss')
                        (loss*(len(idx)/effective)).backward()
                        losses.append(float(loss.detach().cpu()))
                    torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip'])
                    optimizer.step()
                    position+=effective
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    if batch<=16:
                        raise
                    batch//=2
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Same rows and effective batch; failed gradients discarded.
                    write_json(directory/'batch_adjustment.json',{'physical_batch':batch,'effective_batch':t['effective_batch_size'],'reason':'CUDA OOM'})
                if position % (20*t['effective_batch_size'])==0:
                    save_torch(latest,checkpoint())
                peak_rss=max(peak_rss,psutil.Process().memory_info().rss)
            if status=='TIME_LIMIT':
                break
            p=predict_tabm_model(model,dev.num_B,dev.cat_B,batch,device)
            score=validation_scores(dev.y_B,p)
            history.append({'epoch':epoch,'B':score,'mean_train_loss':float(np.mean(losses)) if losses else None,'seconds':time.monotonic()-start})
            if score['roc_auc']>best_auc:
                best_auc,best_epoch,stale=score['roc_auc'],epoch,0
                save_torch(directory/'best.pt',checkpoint())
            else:
                stale+=1
            print(f'TabM {candidate["id"]} seed={seed} epoch={epoch} B-AUC={score["roc_auc"]:.6f}',flush=True)
            epoch+=1
            permutation=None
            position=0
            if stale>=t['patience']:
                status='EARLY_STOPPED'
                break
            model.train()
    except BaseException:
        save_torch(latest,checkpoint())
        raise
    eligible=(directory/'best.pt').exists()
    score=None
    if eligible:
        state=torch.load(directory/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(state['model'])
        p=predict_tabm_model(model,dev.num_B,dev.cat_B,batch,device)
        score=validation_scores(dev.y_B,p)
        # Independent reconstruction verifies checkpoint round-trip.
        restored=make_tabm(bins,cardinalities,cfg,candidate['dropout'],device)
        restored.load_state_dict(state['model'])
        pp=predict_tabm_model(restored,dev.num_B[:64],dev.cat_B[:64],batch,device)
        assert np.allclose(p[:64],pp,atol=1e-6,rtol=1e-5)
        del restored
    result={'family':'TabM','seed':seed,'config':candidate,'B':score,'status':status,'eligible':eligible,
            'converged':status=='EARLY_STOPPED','seconds':time.monotonic()-start,'best_epoch':best_epoch,'epochs_completed':len(history),
            'checkpoint':str((directory/'best.pt').resolve()) if eligible else None,
            'checkpoint_sha256':sha256(directory/'best.pt') if eligible else None,
            'physical_batch':batch,'effective_batch':t['effective_batch_size'],'resumed':resumed,
            'peak_rss_bytes':peak_rss,'gpu_peak_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,'device':device}
    write_json(directory/'curves.json',history)
    write_json(directory/'result.json',result)
    del model,optimizer
    gc.collect()
    if device=='cuda':
        torch.cuda.empty_cache()
    return result


def predict(record,num,cat,threads=4):
    if sha256(record['checkpoint'])!=record['checkpoint_sha256']:
        raise Blocked('Checkpoint hash mismatch')
    if record['family']=='LightGBM':
        p=lgb.Booster(model_file=record['checkpoint']).predict(lgb_matrix(num,cat),num_threads=threads)
    else:
        device='cuda' if torch.cuda.is_available() else 'cpu'
        state=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
        model=make_tabm(state['bins'],state['cardinalities'],state['config'],state['candidate']['dropout'],device)
        model.load_state_dict(state['model'])
        p=predict_tabm_model(model,num,cat,record['physical_batch'],device)
        del model
        gc.collect()
        if device=='cuda':
            torch.cuda.empty_cache()
    check_probabilities(p)
    return p
