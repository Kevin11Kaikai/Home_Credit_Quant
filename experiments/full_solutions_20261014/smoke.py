"""One short full-width training smoke per model, using A only."""
import gc,time
import numpy as np,pandas as pd,torch
from common import *
from data import raw,labels,DenseAdapter,frame
from models import build,forward,loss

def main():
    if (HERE/'smoke_receipt.json').exists():print('REUSE_SMOKE',flush=True);return
    from masamlp.data.preprocessing import TabularPreprocessor
    from catstat import TargetEncoder
    x,s=raw();y,_=labels();ix=np.sort(np.random.default_rng(42).choice(len(y),1024,replace=False));a=np.array(x[ix]);y=y[ix];results=[]
    df=frame(a,s);enc=TargetEncoder(cols=s['names'],target_type='binary',numeric='direct',cv=5,random_state=42);te=np.asarray(enc.fit_transform(df,y),np.float32)
    aug=pd.concat([df,pd.DataFrame(te,columns=[f'te_{i}' for i in range(te.shape[1])])],axis=1)
    pre=TabularPreprocessor(categorical_features=s['categorical']);ns,cs=pre.fit_transform(aug)
    for kind in ['dense','excel','s6e7']:
        seed_all(42);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        if kind=='dense':
            xx=DenseAdapter().fit(a,s).transform(a);cc=None;meta=dict(n_num=xx.shape[1]);batch=256
        elif kind=='excel':
            z=np.nan_to_num(a,nan=0);z=(z-z.mean(0))/(z.std(0)+1e-5);xx=np.column_stack([z,np.zeros((len(z),300))]).astype(np.float32);cc=None;meta=dict(n_num=xx.shape[1],mi_weights=[1/xx.shape[1]]*xx.shape[1]);batch=32
        else:xx=ns;cc=cs;meta=dict(n_num=ns.shape[1],cat_cardinalities=pre.cat_cardinalities_);batch=128
        model=build(kind,meta).cuda();xt=torch.tensor(xx[:batch],device='cuda');ct=None if cc is None else torch.tensor(cc[:batch],device='cuda',dtype=torch.long);yt=torch.tensor(y[:batch],device='cuda',dtype=torch.float32);mi=torch.tensor(meta['mi_weights'],device='cuda') if kind=='excel' else None
        model.eval()
        with torch.no_grad():
            full=forward(model,kind,xt,ct).float().sigmoid()
            with torch.autocast('cuda',dtype=torch.bfloat16):mixed=forward(model,kind,xt,ct).float().sigmoid()
        err=float((full-mixed).abs().max());assert err<.02 and torch.isfinite(mixed).all()
        opt=torch.optim.AdamW(model.parameters(),lr=.001);times=[];model.train()
        for i in range(4):
            torch.cuda.synchronize();tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):v=loss(model,kind,xt,ct,yt,mi)
            assert torch.isfinite(v);v.backward();opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-tick)
        rate=batch/np.median(times[1:]);r=dict(kind=kind,full_width_inputs=meta['n_num']+len(meta.get('cat_cardinalities',[])),batch=batch,samples_per_second=float(rate),peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,amp_max_probability_difference=err,only_A_rows=True,status='PASS')
        results.append(r);print('SMOKE',r,flush=True);del model,opt,xt,ct,yt,mi;gc.collect()
    write(HERE/'smoke_receipt.json',dict(results=results,training_indices_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),gpu=torch.cuda.get_device_name(0),finished_utc=now()))
if __name__=='__main__':main()
