"""Only A/B can be loaded. No C/D feature or label interface exists here."""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import *

def build_raw(variant='cat'):
    out=HERE/'derived'/('raw_'+variant);out.mkdir(parents=True,exist_ok=True)
    if saved(out):return read(out/'receipt.json')
    schema=read(PREVIOUS/'derived'/variant/'schema.json')
    n=schema['n_A']+schema['n_B'];names=schema['numeric']+schema['categorical'];cats=set(schema['categorical'])
    assert not any(n in cats or n in schema['numeric'] for n in ['target','case_id','row_id','WEEK_NUM','date_decision','index_in_full','x_meta'])
    x=np.lib.format.open_memmap(out/'raw.npy',mode='w+',shape=(n,len(names)),dtype=np.float64)
    ci={s:i for i,s in enumerate(names)};vocab={}
    for item in schema['feature_tables']:
        path=PREVIOUS/item['path'];assert sha(path)==item['sha256']
        table=pq.read_table(path)
        for name in table.column_names:
            if name in cats:
                s=table[name].to_pandas();seen=sorted(s.iloc[:schema['n_A']].dropna().astype(str).unique())
                v={c:i+2 for i,c in enumerate(seen)};vocab[name]=v
                # Codes are equality-preserving identifiers; every model fits its own vocabulary.
                x[:,ci[name]]=s.astype('string').map(v).fillna(0).where(~s.isna(),1).to_numpy(float)
            else:
                a=table[name].to_numpy(zero_copy_only=False).astype(float);a[~np.isfinite(a)]=np.nan;x[:,ci[name]]=a
        del table;gc.collect()
    x.flush();del x
    receipt=dict(variant=variant,n_A=schema['n_A'],n_B=schema['n_B'],numeric=schema['numeric'],categorical=schema['categorical'],names=names,vocab=vocab,missing_code=1,unknown_code=0,source_schema_sha256=sha(PREVIOUS/'derived'/variant/'schema.json'),feature_selection='Inherited A-only missingness/cardinality filters from prior feature experiment, NOT independently fitted inside OOF folds; OOF is development only.',files=seal(out,['raw.npy']),created_utc=now())
    write(out/'receipt.json',receipt);print('RAW_READY',variant,n,len(names),flush=True);return receipt

def labels():return np.load(PREVIOUS/'derived/y_A.npy'),np.load(PREVIOUS/'derived/y_B.npy')
def mapping():return pq.read_table(PREVIOUS/'derived/mapping_AB.parquet').to_pandas()
def raw(variant='cat'):
    root=HERE/'derived'/('raw_'+variant);s=read(root/'receipt.json')
    return np.load(root/'raw.npy',mmap_mode='r'),s
def frame(a,s):
    df=pd.DataFrame(a,columns=s['names'])
    for n in s['categorical']:df[n]=df[n].astype('int64').astype(str)
    return df
def folds(kind):
    from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
    y,_=labels();dest=HERE/'derived'/('folds_'+kind);dest.mkdir(exist_ok=True)
    if not (dest/'receipt.json').exists():
        splitter=StratifiedKFold(7,shuffle=True,random_state=42) if kind=='s6e7' else StratifiedGroupKFold(5,shuffle=True,random_state=42)
        groups=None if kind=='s6e7' else mapping().iloc[:len(y)]['WEEK_NUM'].to_numpy()
        files=[]
        for i,(tr,va) in enumerate(splitter.split(np.zeros(len(y)),y,groups)):
            assert not np.intersect1d(tr,va).size
            if groups is not None:assert not set(groups[tr])&set(groups[va])
            name=f'fold_{i}.npz';np.savez(dest/name,train=tr,valid=va);files.append(name)
        write(dest/'receipt.json',dict(kind=kind,files=seal(dest,files),scope='A only; original temporal B remains later development evaluation, C/D untouched'))
    saved(dest)
    return [np.load(p) for p in sorted(dest.glob('fold_*.npz'))]

class DenseAdapter:
    """Fold-only medians/scales and categorical one-hot; reconstruction choice."""
    def fit(self,a,s):
        from sklearn.preprocessing import OneHotEncoder
        self.n=len(s['numeric']);num=np.asarray(a[:,:self.n])
        with __import__('warnings').catch_warnings():
            __import__('warnings').simplefilter('ignore',RuntimeWarning);self.median=np.nanmedian(num,axis=0)
        self.median[~np.isfinite(self.median)]=0
        num=np.where(np.isfinite(num),num,self.median);self.mean=num.mean(0);self.scale=num.std(0);self.scale[self.scale==0]=1
        self.enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32).fit(a[:,self.n:]);return self
    def transform(self,a):
        n=np.asarray(a[:,:self.n]);n=((np.where(np.isfinite(n),n,self.median)-self.mean)/self.scale).astype(np.float32)
        return np.column_stack([n,self.enc.transform(a[:,self.n:])]).astype(np.float32)

def prepare_excel():
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.feature_selection import mutual_info_classif
    from category_encoders import CatBoostEncoder
    import joblib
    dest=HERE/'derived/excel';dest.mkdir(exist_ok=True)
    if saved(dest):return read(dest/'receipt.json')
    tick=time.perf_counter();x,s=raw();n=s['n_A'];k=len(s['numeric']);y,_=labels();dt=mapping()['date_decision'].to_numpy()
    flag=~np.isfinite(x[:n,:k]);groups={}
    for j in range(k):
        if flag[:,j].any():
            h=hashlib.sha256(np.packbits(flag[:,j]).tobytes()).hexdigest();groups.setdefault(h,[]).append(j)
    flag_groups=list(groups.values());del flag
    with __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore',RuntimeWarning);med=np.nanmedian(x[:n,:k],axis=0)
    med[~np.isfinite(med)]=0
    a=np.where(np.isfinite(x[:n,:k]),x[:n,:k],med).astype(np.float32)
    qt=QuantileTransformer(n_quantiles=1024,output_distribution='normal',subsample=100000,random_state=42).fit(a);del a
    cuts=np.linspace(0,n,6,dtype=int)
    for j in range(1,5):cuts[j]=np.searchsorted(dt[:n],dt[cuts[j]],side='left')
    codes=pd.DataFrame(x[:,k:].astype(np.int64).astype(str),columns=s['categorical'])
    te=np.full((len(x),len(s['categorical'])),.5,dtype=np.float32)
    for j in range(1,5):
        start,end=cuts[j:j+2];assert dt[start-1]<dt[start]
        enc=CatBoostEncoder(cols=s['categorical'],a=1.,random_state=42).fit(codes.iloc[:start],y[:start]);te[start:end]=enc.transform(codes.iloc[start:end]).to_numpy(np.float32)
    enc=CatBoostEncoder(cols=s['categorical'],a=1.,random_state=42).fit(codes.iloc[:n],y);te[n:]=enc.transform(codes.iloc[n:]).to_numpy(np.float32)
    del codes
    names=s['numeric']+[f'missing_group_{j}' for j in range(len(flag_groups))]
    names += [v+suffix for v in s['categorical'] for suffix in ['__missing','__unknown']]+[v+'__ordered_te' for v in s['categorical']]
    buf=np.lib.format.open_memmap(dest/'unsorted.npy',mode='w+',dtype=np.float32,shape=(len(x),len(names)))
    for off in range(0,len(x),8192):
        chunk=np.asarray(x[off:off+8192]);missing=~np.isfinite(chunk[:,:k])
        parts=[qt.transform(np.where(missing,med,chunk[:,:k])).astype(np.float32),np.column_stack([missing[:,g].any(1) for g in flag_groups]).astype(np.float32)]
        parts += [(chunk[:,k+j:k+j+1]==v).astype(np.float32) for j in range(len(s['categorical'])) for v in [1,0]]
        parts += [te[off:off+len(chunk)]];buf[off:off+len(chunk)]=np.column_stack(parts)
    buf.flush();del te
    ix=np.random.default_rng(42).choice(n,min(20000,n),replace=False)
    mi=mutual_info_classif(np.asarray(buf[ix]),y[ix],random_state=42,discrete_features=False,n_neighbors=3)
    assert np.isfinite(mi).all() and mi.sum()>0;order=np.lexsort((np.arange(len(mi)),-mi))
    dst=np.lib.format.open_memmap(dest/'x.npy',mode='w+',dtype=np.float32,shape=buf.shape)
    for off in range(0,len(buf),8192):dst[off:off+8192]=buf[off:off+8192,order]
    dst.flush();del dst,buf
    joblib.dump(dict(median=med,quantile=qt,encoder=enc,flag_groups=flag_groups,order=order,cuts=cuts),dest/'preprocessor.joblib');np.save(dest/'mi_rows.npy',ix)
    r=dict(names=[names[j] for j in order],mi_weights=(mi[order]/mi.sum()).tolist(),n_A=n,n_B=s['n_B'],chronological_cuts=cuts.tolist(),source=sha(HERE/'derived/raw_cat/receipt.json'),source_adaptation='Credit application history aggregates replace unavailable fraud card/address/UID fields; no cross-customer identity features claimed.',seconds=time.perf_counter()-tick,files=seal(dest,['x.npy','preprocessor.joblib','mi_rows.npy']))
    write(dest/'receipt.json',r);print('EXCEL_INPUT_READY',len(names),round(r['seconds'],1),flush=True);return r

def prepare_s6e7(fold):
    import joblib
    from catstat import TargetEncoder
    from masamlp.data.preprocessing import TabularPreprocessor
    dest=HERE/'derived'/f's6e7_{fold}';dest.mkdir(exist_ok=True)
    if saved(dest):return read(dest/'receipt.json')
    tick=time.perf_counter();x,s=raw();y,_=labels();idx=folds('s6e7')[fold];tr,va=idx['train'],idx['valid'];n=len(y)
    # Encoder is processed column-wise to cap RAM. catstat's same fixed internal
    # row-fold assignment is used on every column, exactly as the joint encoder.
    raw_cols=s['names'];nn=len(s['numeric']);n_te=len(raw_cols)
    arrays={};files=[]
    for part,rows in [('train',tr),('valid',va),('B',np.arange(n,len(x)))]:
        path=f'{part}_aug.npy';arrays[part]=np.lib.format.open_memmap(dest/path,mode='w+',dtype=np.float64,shape=(len(rows),len(raw_cols)+n_te));arrays[part][:,:len(raw_cols)]=x[rows];files.append(path)
    encoders=[]
    for j,col in enumerate(raw_cols):
        f=pd.DataFrame({col:np.asarray(x[:,j])})
        enc=TargetEncoder(cols=[col],stats=('mean',),target_type='binary',smooth='auto',numeric='direct',cv=5,random_state=42)
        arrays['train'][:,len(raw_cols)+j]=np.asarray(enc.fit_transform(f.iloc[tr],y[tr]),dtype=np.float32).reshape(-1)
        for part,rows in [('valid',va),('B',np.arange(n,len(x)))]:arrays[part][:,len(raw_cols)+j]=np.asarray(enc.transform(f.iloc[rows]),dtype=np.float32).reshape(-1)
        encoders.append(enc)
        if j%100==0:print('TE',fold,j,'/',len(raw_cols),flush=True)
    names=raw_cols+[f'te_{j}' for j in range(n_te)];cat_idx=list(range(nn,len(raw_cols)))
    # Official preprocessor is fit only on the outer training fold and its
    # cross-fitted target-encoded rows. Its OOV slot remains explicit.
    pre=TabularPreprocessor(numeric_scaler='quantile',cat_encoding='embedding',categorical_features=cat_idx)
    pre.fit(pd.DataFrame(arrays['train'],columns=names))
    for part,a in arrays.items():
        ns,cs=pre.transform(pd.DataFrame(a,columns=names))
        np.save(dest/f'{part}_num.npy',ns);np.save(dest/f'{part}_cat.npy',cs);a.flush()
    joblib.dump(pre,dest/'preprocessor.joblib');joblib.dump(encoders,dest/'target_encoders.joblib')
    files=[f'{p}_{t}.npy' for p in ['train','valid','B'] for t in ['num','cat']]+['preprocessor.joblib','target_encoders.joblib']
    r=dict(fold=fold,n_train=len(tr),n_valid=len(va),n_B=s['n_B'],n_num=ns.shape[1],cat_cardinalities=pre.cat_cardinalities_,raw_features=len(raw_cols),TE_features=n_te,target='Binary task adaptation: one target mean and one sigmoid logit, instead of three class means/softmax.',fit_rows_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),files=seal(dest,files),seconds=time.perf_counter()-tick)
    write(dest/'receipt.json',r);print('S6E7_INPUT_READY',fold,r['n_num'],len(r['cat_cardinalities']),round(r['seconds'],1),flush=True);return r

