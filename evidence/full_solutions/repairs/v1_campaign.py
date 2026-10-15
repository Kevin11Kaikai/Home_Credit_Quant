"""Executable fixed campaign. Complete artifacts are reused, never refit silently."""
from __future__ import annotations
import argparse,gc,sys,traceback,importlib.metadata
import joblib,numpy as np,pandas as pd
from common import *
from data import raw,build_raw,folds,labels,mapping,DenseAdapter,prepare_excel,prepare_s6e7
from models import train_neural,build,predict

def freeze():
    path=HERE/'PROTOCOL_LOCK.json'
    code={n:sha(HERE/n) for n in ['common.py','data.py','models.py','campaign.py']}
    if path.exists():
        lock=read(path);assert lock['code']==code,'Code changed after freeze';assert lock['config_sha256']==sha(HERE/'config.json')
        for p,h in lock['protected_artifacts'].items():assert sha(p)==h,p
        for p,h in lock['data'].items():assert sha(p)==h,p
        for p,h in lock['vendor_hashes'].items():assert sha(HERE/p)==h,p
        return lock
    cfg=read(HERE/'config.json')
    for v in ['cat','lgb']:saved(HERE/'derived'/('raw_'+v))
    manifest=dict(created_utc=now(),config_sha256=sha(HERE/'config.json'),code=code,protected_artifacts=protected(),data={str(p):sha(p) for p in [PREVIOUS/'derived/y_A.npy',PREVIOUS/'derived/y_B.npy',PREVIOUS/'derived/mapping_AB.parquet',HERE/'derived/raw_cat/receipt.json',HERE/'derived/raw_lgb/receipt.json']},packages={k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','lightgbm','catboost','masamlp','catstat','lightautoml','category-encoders']},evaluation='Post-selection exploratory A-OOF/B development only; independent original C/D untouched.',source_hashes={str(p.relative_to(HERE)):sha(p) for p in (HERE/'sources').rglob('*') if p.is_file()},vendor_hashes={str(p.relative_to(HERE)):sha(p) for p in (HERE/'vendor').glob('*.py')})
    write(path,manifest);write(HERE/'requirements.lock.json',manifest['packages']);return manifest

def champion_tree(kind,fold,cfg,budget):
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    root=HERE/'runs'/f'champion_{kind}_fold{fold}';root.mkdir(parents=True,exist_ok=True)
    if saved(root):return read(root/'receipt.json')
    x,s=raw('lgb' if kind=='lgb' else 'cat');idx=folds('champion')[fold];tr,va=idx['train'],idx['valid'];y,yb=labels();n=len(y);k=len(s['numeric'])
    config=cfg['lightgbm'] if kind=='lgb' else cfg['catboost'];tick=time.perf_counter()
    def df(rows):
        out=pd.DataFrame(np.asarray(x[rows],dtype=np.float32),columns=s['names'])
        for j,col in enumerate(s['categorical']):
            known=np.unique(x[tr,k+j]);v=np.asarray(x[rows,k+j],dtype=np.int64);v=np.where(np.isin(v,known),v,0)
            out[col]=pd.Categorical(v,categories=np.r_[0,known[known!=0]]) if kind=='lgb' else v.astype(str)
        return out
    train=df(tr);valid=df(va);later=df(np.arange(n,len(x)))
    if kind=='lgb':
        model=lgb.train(config,lgb.Dataset(train,label=y[tr],categorical_feature=s['categorical']),num_boost_round=cfg['lightgbm_rounds'],valid_sets=[lgb.Dataset(valid,label=y[va],categorical_feature=s['categorical'])],callbacks=[lgb.early_stopping(cfg['early_stopping'],verbose=False)])
        pv=model.predict(valid);pb=model.predict(later);model.save_model(str(root/'model.txt'));reload=lgb.Booster(model_file=str(root/'model.txt'));np.testing.assert_allclose(reload.predict(later.iloc[:257]),pb[:257],rtol=0,atol=1e-12);model_file='model.txt';iteration=model.best_iteration
    else:
        model=CatBoostClassifier(**config);model.fit(train,y[tr],cat_features=s['categorical'],eval_set=(valid,y[va]),early_stopping_rounds=cfg['early_stopping']);pv=model.predict_proba(valid)[:,1];pb=model.predict_proba(later)[:,1];model.save_model(str(root/'model.cbm'));reload=CatBoostClassifier().load_model(str(root/'model.cbm'));np.testing.assert_allclose(reload.predict_proba(later.iloc[:257])[:,1],pb[:257],rtol=0,atol=1e-12);model_file='model.cbm';iteration=model.best_iteration_
    np.save(root/'p_valid.npy',pv);np.save(root/'p_B.npy',pb);joblib.dump(dict(numeric=s['numeric'],categorical=s['categorical'],fold_training_rows=tr,category_vocab={col:np.unique(x[tr,k+j]) for j,col in enumerate(s['categorical'])}),root/'preprocessor.joblib')
    r=dict(status='COMPLETE',kind=kind,fold=fold,config=config,seconds=time.perf_counter()-tick,iterations=iteration,metrics=metrics(y[va],pv),B=metrics(yb,pb),selected_checkpoint=model_file,files=seal(root,[model_file,'preprocessor.joblib','p_valid.npy','p_B.npy']),n_train=len(tr),n_valid=len(va),n_B=len(yb),finished_utc=now())
    write(root/'receipt.json',r);print('CHAMPION_FOLD',kind,fold,round(r['seconds'],1),r['B']['roc_auc'],flush=True)
    del train,valid,later,model,reload;gc.collect();budget.beat();return r

def champion_dense(fold,cfg,budget):
    root=HERE/'runs'/f'champion_dense_fold{fold}';root.mkdir(parents=True,exist_ok=True)
    if saved(root) and (root/'B_receipt.json').exists():verify(root,read(root/'B_receipt.json')['files']);return
    x,s=raw();y,yb=labels();idx=folds('champion')[fold];tr,va=idx['train'],idx['valid'];n=len(y)
    if (root/'preprocessor.joblib').exists():pre=joblib.load(root/'preprocessor.joblib')
    else:pre=DenseAdapter().fit(x[tr],s);joblib.dump(pre,root/'preprocessor.joblib')
    tx=pre.transform(x[tr]);vx=pre.transform(x[va]);bx=pre.transform(x[n:]);meta=dict(n_num=tx.shape[1],preprocessor_sha256=sha(root/'preprocessor.joblib'))
    r=train_neural('dense',root,meta,tx,None,y[tr],vx,None,y[va],cfg['dense'],budget)
    import torch
    model=build('dense',meta).cuda();model.load_state_dict(torch.load(root/r['selected_checkpoint'],map_location='cpu',weights_only=False)['model']);pb=predict(model,'dense',bx,batch=1024);np.save(root/'p_B.npy',pb)
    write(root/'B_receipt.json',dict(B=metrics(yb,pb),files=seal(root,['p_B.npy','preprocessor.joblib']),status=r['status']));del model,tx,vx,bx;gc.collect();torch.cuda.empty_cache()

def score_postprocess(p,week,subtract=.03):
    # Competition score only. Never named calibrated probability or used in EL.
    week=np.asarray(week);early=week < week.min()+(week.max()-week.min())*.5
    score=np.array(p,copy=True);score[early]=np.maximum(0,score[early]-subtract);return score

def champion_ensemble():
    y,yb=labels();cfg=read(HERE/'config.json')['champion'];root=HERE/'results/champion';root.mkdir(parents=True,exist_ok=True)
    if saved(root):return
    pb=np.zeros(len(yb));oof=np.zeros(len(y));coverage=np.zeros(len(y),dtype=int);rows=[]
    for kind,weight in cfg['weights'].items():
        mean=np.zeros(len(yb));oo=np.zeros(len(y));counts=np.zeros(len(y),dtype=int)
        for f,ix in enumerate(folds('champion')):
            d=HERE/'runs'/f'champion_{kind}_fold{f}';r=saved(d)
            if not r or r['status']!='COMPLETE':raise RuntimeError('Champion reconstruction incomplete: '+d.name)
            pv=np.load(d/'p_valid.npy');pred=np.load(d/'p_B.npy');mean+=pred/cfg['folds'];oo[ix['valid']]=pv;counts[ix['valid']]+=1
        assert np.all(counts==1);pb+=weight*mean;oof+=weight*oo;np.save(root/f'{kind}_B.npy',mean);rows.append(dict(model='Champion reconstruction '+kind,split='B',**metrics(yb,mean)))
    np.save(root/'ensemble_B.npy',pb);np.save(root/'ensemble_A_oof.npy',oof)
    weeks=mapping().iloc[len(y):]['WEEK_NUM'].to_numpy();hacked=score_postprocess(pb,weeks);np.save(root/'score_hack_B.npy',hacked)
    rows.append(dict(model='Champion reconstruction blend',split='B',**metrics(yb,pb)))
    # Log loss of the heuristic is labeled score diagnostic, not a calibrated PD.
    rows.append(dict(model='Champion reconstruction + score hack (not PD)',split='B',**metrics(yb,hacked)))
    pd.DataFrame(rows).to_csv(root/'metrics.csv',index=False)
    write(root/'receipt.json',dict(status='THREE_FAMILY_RECONSTRUCTION_COMPLETE',exact_champion_status='BLOCKED_DEPENDENCY',weights=cfg['weights'],scope=cfg['status_boundary'],files=seal(root,[p.name for p in root.iterdir() if p.is_file()]),finished_utc=now()))
    print('CHAMPION_RECONSTRUCTION_COMPLETE',flush=True)

def excel(cfg,budget):
    r=prepare_excel();x=np.load(HERE/'derived/excel/x.npy',mmap_mode='r');y,yb=labels();n=len(y);meta=dict(n_num=x.shape[1],mi_weights=r['mi_weights'],preprocessor_receipt_sha256=sha(HERE/'derived/excel/receipt.json'))
    root=HERE/'runs/excel';result=train_neural('excel',root,meta,x[:n],None,y,x[n:],None,yb,cfg,budget)
    # valid means B for this family only; explicitly preserve this identity.
    write(root/'evaluation_scope.json',dict(split='B',checkpoint_selection='B average precision, matching the requested fraud source',independent_test=False))
    return result

def s6e7(cfg,budget):
    import torch
    y,yb=labels()
    for f,idx in enumerate(folds('s6e7')):
        r=prepare_s6e7(f);src=HERE/'derived'/f's6e7_{f}'
        arrays={f'{p}_{t}':np.load(src/f'{p}_{t}.npy',mmap_mode='r') for p in ['train','valid','B'] for t in ['num','cat']}
        meta=dict(n_num=r['n_num'],cat_cardinalities=r['cat_cardinalities'],preprocessing_receipt_sha256=sha(src/'receipt.json'))
        for m,seed in enumerate(cfg['member_seeds']):
            root=HERE/'runs'/f's6e7_fold{f}_member{m}'
            if saved(root) and (root/'B_receipt.json').exists():verify(root,read(root/'B_receipt.json')['files']);continue
            member_cfg=cfg|{'seed':seed}
            result=train_neural('s6e7',root,meta,arrays['train_num'],arrays['train_cat'],y[idx['train']],arrays['valid_num'],arrays['valid_cat'],y[idx['valid']],member_cfg,budget)
            model=build('s6e7',meta,seed).cuda();model.load_state_dict(torch.load(root/result['selected_checkpoint'],map_location='cpu',weights_only=False)['model']);p=predict(model,'s6e7',arrays['B_num'],arrays['B_cat'],cfg['inference_batch']);np.save(root/'p_B.npy',p);write(root/'B_receipt.json',dict(files=seal(root,['p_B.npy']),status=result['status'],B=metrics(yb,p)))
            del model;gc.collect();torch.cuda.empty_cache()
            if result['status']!='COMPLETE':return
    s6e7_ensemble(cfg)

def s6e7_ensemble(cfg):
    from sklearn.metrics import balanced_accuracy_score
    y,yb=labels();root=HERE/'results/s6e7';root.mkdir(parents=True,exist_ok=True)
    if saved(root):return
    pb=np.zeros(len(yb));oof=np.zeros(len(y));counts=np.zeros(len(y),int);members=[]
    for f,idx in enumerate(folds('s6e7')):
        pv=np.zeros(len(idx['valid']))
        for m,seed in enumerate(cfg['member_seeds']):
            d=HERE/'runs'/f's6e7_fold{f}_member{m}';r=saved(d)
            assert r['status']=='COMPLETE' and r['completed_epochs']==16
            pv+=np.load(d/'p_valid.npy')/4;pb+=np.load(d/'p_B.npy')/28;members.append(dict(fold=f,seed=seed,checkpoint=sha(d/r['selected_checkpoint'])))
        oof[idx['valid']]=pv;counts[idx['valid']]+=1
    assert np.all(counts==1)
    np.save(root/'ensemble_B.npy',pb);np.save(root/'ensemble_A_oof.npy',oof)
    # Retain the source classification postprocessing separately. OOF tuning is
    # exploratory, and the corrected class decisions never replace PD proxies.
    prior=np.array([1-y.mean(),y.mean()]);proba=np.column_stack([1-oof,oof]);scores={str(b):float(balanced_accuracy_score(y,(proba/prior**b).argmax(1))) for b in cfg['beta_grid_classification_only']};best=max(scores,key=scores.get)
    decision=(np.column_stack([1-pb,pb])/prior**float(best)).argmax(1);np.save(root/'B_classification_only.npy',decision)
    write(root/'classification_postprocess.json',dict(beta_grid=scores,selected_beta=float(best),B_balanced_accuracy=float(balanced_accuracy_score(yb,decision)),scope='Class decisions only; raw probabilities retained for probability scores.'))
    pd.DataFrame([dict(model='S6E7 transfer 7x4',split='B',**metrics(yb,pb))]).to_csv(root/'metrics.csv',index=False)
    write(root/'receipt.json',dict(status='COMPLETE_7_FOLDS_4_MEMBERS',members=members,epochs_each=16,files=seal(root,[p.name for p in root.iterdir() if p.is_file()]),finished_utc=now()))

def run(family):
    lock=freeze();cfg=read(HERE/'config.json');budget=Budget();start=time.perf_counter()
    families=['champion','excel','s6e7'] if family=='all' else [family]
    for kind in families:
        write(HERE/'campaign_state.json',dict(status='RUNNING',family=kind,pid=os.getpid(),started_utc=now()))
        try:
            if kind=='champion':
                for name in ['lgb','cat','dense']:
                    budget.start(name)
                    try:
                        for f in range(cfg['champion']['folds']):
                            if name=='dense':champion_dense(f,cfg['champion'],budget)
                            else:champion_tree(name,f,cfg['champion'],budget)
                            if budget.used(name)>=cfg['limits'][name+'_seconds']:raise RuntimeError('BUDGET_LIMIT '+name)
                    finally:budget.stop()
                champion_ensemble()
            else:
                budget.start(kind)
                try:(excel if kind=='excel' else s6e7)(cfg[kind],budget)
                finally:budget.stop()
        except Exception as e:
            write(HERE/f'{kind}_failure.json',dict(status='FAILED_OR_BLOCKED',error=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=now()));print(traceback.format_exc(),flush=True)
            # Independent authorized methods proceed; failure is never PASS.
        if time.perf_counter()-start>=cfg['limits']['wall_seconds']:break
    for p,h in lock['protected_artifacts'].items():assert sha(p)==h,p
    write(HERE/'campaign_state.json',dict(status='STOPPED_CHECK_ACCEPTANCE',family=family,pid=os.getpid(),finished_utc=now()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','freeze','run']);p.add_argument('--family',choices=['all','champion','excel','s6e7'],default='all');args=p.parse_args()
    if args.command=='prepare':
        for v in ['lgb','cat']:build_raw(v)
        for v in ['champion','s6e7']:folds(v)
    elif args.command=='freeze':freeze()
    else:run(args.family)
