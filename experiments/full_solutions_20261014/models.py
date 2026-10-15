"""Official model implementations; local training adds accumulation and recovery."""
from __future__ import annotations
import types,math,gc
import torch
from torch.nn import functional as F
import numpy as np
from common import *

def excel_attention_sdpa(self,x_q,x_kv,key_compression,value_compression):
    # Algebraic replacement of the upstream dense attention matrix. Keep its
    # finite -1e4 mask (rather than changing to an infinite causal mask).
    assert key_compression is None and value_compression is None
    b,q_len,_=x_q.shape;k_len=x_kv.shape[1];h=self.n_heads
    q=self.W_q(x_q).reshape(b,q_len,h,-1).transpose(1,2)
    k=self.W_k(x_kv).reshape(b,k_len,h,-1).transpose(1,2)
    v=self.W_v(x_kv).reshape(b,k_len,h,-1).transpose(1,2)
    mask=(torch.arange(k_len,device=q.device)[None,:]>torch.arange(q_len,device=q.device)[:,None]).to(q.dtype)*-1e4
    value=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=self.dropout.p if self.training and self.dropout is not None else 0.)
    out=value.transpose(1,2).reshape(b,q_len,-1)
    return self.W_out(out) if self.W_out is not None else out

def build(kind,meta,seed=42):
    seed_all(seed)
    if kind=='s6e7':
        from masamlp.models import build_model
        return build_model('ft_transformer',dict(d_block=128,n_blocks=2,attention_n_heads=8,n_frequencies=24,sigma=.1),n_num=meta['n_num'],cat_cardinalities=meta['cat_cardinalities'],out_dim=1,num_embedding='plr-lite')
    if kind=='dense':
        from lightautoml.ml_algo.torch_based.nn_models import DenseLightModel
        return DenseLightModel(n_in=meta['n_num'])
    if kind=='excel':
        from vendor.excel import ExcelFormer,MultiheadAttention
        m=ExcelFormer(d_numerical=meta['n_num'],categories=None,token_bias=True,n_layers=3,d_token=128,n_heads=8,attention_dropout=.3,ffn_dropout=0.,residual_dropout=0.,prenormalization=True,kv_compression=None,kv_compression_sharing=None,d_out=1,init_scale=.01)
        for module in m.modules():
            if isinstance(module,MultiheadAttention):module.forward=types.MethodType(excel_attention_sdpa,module)
        return m
    raise ValueError(kind)

def forward(model,kind,x,c=None):return model(x) if kind=='dense' else model(x,c)

def predict(model,kind,x,c=None,batch=128):
    model.eval();pieces=[]
    with torch.inference_mode():
        for off in range(0,len(x),batch):
            xn=torch.as_tensor(np.asarray(x[off:off+batch]).copy(),device='cuda',dtype=torch.float32)
            xc=None if c is None else torch.as_tensor(np.asarray(c[off:off+batch]).copy(),device='cuda',dtype=torch.long)
            with torch.autocast('cuda',dtype=torch.bfloat16):z=forward(model,kind,xn,xc)
            pieces.append(z.float().reshape(-1).sigmoid().cpu().numpy())
    p=np.concatenate(pieces);assert np.isfinite(p).all();return p

def loss(model,kind,x,c,y,mi=None):
    if kind=='excel':
        z,mask,perm=model(x,None,mixup=True,beta=.5,mtype='feat_mix')
        lam=(mask.float()*mi).sum(1)
        a=F.binary_cross_entropy_with_logits(z.reshape(-1).float(),y,reduction='none')
        b=F.binary_cross_entropy_with_logits(z.reshape(-1).float(),y[perm],reduction='none')
        return (lam*a+(1-lam)*b).mean()
    return F.binary_cross_entropy_with_logits(forward(model,kind,x,c).reshape(-1).float(),y)

def train_neural(kind,root,meta,train_x,train_c,y,valid_x,valid_c,valid_y,cfg,budget):
    """No C/D parameters. B is passed only for the explicitly selected Excel run.

    S6E7 receives A's outer heldout fold, is fixed-epoch and does not select on
    this label stream. Epoch and mid-epoch checkpoints preserve optimizer/RNG.
    """
    root.mkdir(parents=True,exist_ok=True)
    if saved(root):return read(root/'receipt.json')
    lock=read(HERE/'PROTOCOL_LOCK.json');code={n:sha(HERE/n) for n in ['common.py','data.py','models.py','campaign.py']}
    if code!=lock['code']:raise RuntimeError('Code changed after freeze')
    limits=read(HERE/'config.json')['limits']
    wall_deadline=datetime.fromisoformat(lock['created_utc']).timestamp()+limits['wall_seconds']
    meta_hash=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    seed_all(cfg['seed']);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    model=build(kind,meta,cfg['seed']).cuda()
    if kind=='s6e7':
        with torch.no_grad():model.output_layer.bias.fill_(float(np.log(y.mean()/(1-y.mean()))))
    seed_all(cfg['seed']) # Official Masa Trainer re-seeds after model construction.
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    gen=torch.Generator().manual_seed(cfg['seed'])
    x=torch.as_tensor(np.asarray(train_x).copy(),device='cuda',dtype=torch.float32)
    c=None if train_c is None else torch.as_tensor(np.asarray(train_c).copy(),device='cuda',dtype=torch.long)
    yt=torch.as_tensor(y,device='cuda',dtype=torch.float32)
    mi=None if kind!='excel' else torch.tensor(meta['mi_weights'],device='cuda',dtype=torch.float32)
    history=[];epoch=1;offset=0;order=None;best=-1.;bad=0;total=0.;count=0;elapsed=0.;best_epoch=0
    if (root/'last.pt').exists():
        cp=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        assert cp['config']==cfg and cp['meta_hash']==meta_hash
        if cp['code']!=code:
            permitted=False
            for item in lock.get('repairs',[]):
                record=HERE/item['path'];assert sha(record)==item['sha256'];repair=read(record)
                key=str((root/'last.pt').relative_to(HERE))
                if repair['old_code']==cp['code'] and repair['permitted_old_artifacts'].get(key)==sha(root/'last.pt'):
                    for name,h in repair['permitted_old_artifacts'].items():assert sha(HERE/name)==h,name
                    permitted=True
            assert permitted,'Checkpoint code changed without an artifact-specific repair record'
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer'])
        for state in opt.state.values():
            for key,val in state.items():
                if torch.is_tensor(val) and key!='step':state[key]=val.cuda()
        history=cp['history'];epoch=cp['epoch'];offset=cp['offset'];order=cp['order'];best=cp['best'];bad=cp['bad'];total=cp['total'];count=cp['count'];elapsed=cp['elapsed'];best_epoch=cp['best_epoch'];gen.set_state(cp['generator']);restore_rng(cp['rng'])
        print('RESUMED',root.name,epoch,offset,flush=True)
    def save_progress():
        checkpoint(root/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),generator=gen.get_state(),history=history,epoch=epoch,offset=offset,order=order,best=best,bad=bad,total=total,count=count,elapsed=elapsed,best_epoch=best_epoch,config=cfg,meta_hash=meta_hash,code=code))
    # One full-width real-data smoke is run before formal fits; no AMP fallback
    # or automatic feature dropping is permitted during formal execution.
    started=time.perf_counter();save_tick=started;stop='MAX_EPOCHS'
    # An early-stopped fit can fail during final verification. Finalize its
    # already selected checkpoint without adding another training epoch.
    finalize_existing=order is None and offset==0 and bool(cfg.get('patience')) and epoch>cfg.get('min_epochs',0) and bad>=cfg['patience']
    if finalize_existing:stop='EARLY_STOPPING'
    while epoch<=cfg['epochs'] and not finalize_existing:
        if order is None:
            order=torch.randperm(len(y),generator=gen)
            total=0.;count=0;offset=0
        if kind=='s6e7':lr=cfg['lr']*.5*(1+math.cos(math.pi*(epoch-1)/cfg['epochs']))
        elif kind=='excel':
            lr=cfg['lr']*min(1.,epoch/2)*(.5+.5*math.cos(math.pi*max(0,epoch-2)/38)) if epoch<=16 else 3e-5
            if epoch==17 and offset==0:
                # Published continuation starts from the first-stage best and
                # resets AdamW. Never import a checkpoint from the fraud task.
                cp=torch.load(root/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(cp['model']);opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0.)
        else:lr=cfg['lr']
        for g in opt.param_groups:g['lr']=lr
        model.train();epoch_tick=time.perf_counter();resume_start=offset
        while offset<len(order):
            idx=order[offset:offset+cfg['batch']].to('cuda');opt.zero_grad(set_to_none=True);value=0.
            # DenseLight contains BN. It is not gradient-accumulated; source BN
            # behavior is preserved. Transformers have no BN.
            for ix in idx.split(cfg['microbatch']):
                if kind=='dense' and len(ix)==1:raise RuntimeError('Singleton BatchNorm batch; adjust deterministic batch partition')
                with torch.autocast('cuda',dtype=torch.bfloat16):v=loss(model,kind,x[ix],None if c is None else c[ix],yt[ix],mi)
                if not torch.isfinite(v):raise RuntimeError('Nonfinite loss')
                (v*len(ix)/len(idx)).backward();value+=float(v.detach())*len(ix)
            if cfg.get('grad_clip') is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
            opt.step();total+=value;count+=len(idx);offset+=len(idx)
            if time.perf_counter()-save_tick>=60:
                elapsed+=time.perf_counter()-started;started=time.perf_counter();save_tick=started;budget.beat()
                write(root/'progress.json',dict(status='TRAINING',epoch=epoch,epochs=cfg['epochs'],processed=offset,total=len(y),completed_epochs=len(history),active_seconds=elapsed,updated_utc=now()))
            if (offset//cfg['batch'])%64==0: # recover no more than a few minutes
                save_progress()
            if elapsed+time.perf_counter()-started>=cfg['max_seconds'] or budget.used(kind)>=limits[kind+'_seconds'] or time.time()>=wall_deadline:
                stop='BUDGET_STOP';break
        if stop=='BUDGET_STOP':save_progress();break
        p=predict(model,kind,valid_x,valid_c,cfg['inference_batch']);measured=metrics(valid_y,p)
        monitor=measured[cfg['monitor']]
        if monitor>best:
            best=monitor;best_epoch=epoch;bad=0
            checkpoint(root/'best.pt',dict(model=model.state_dict(),epoch=epoch,config=cfg,meta=meta,code=code));np.save(root/'best_valid.npy',p)
        else:bad+=1
        elapsed+=time.perf_counter()-started;started=time.perf_counter()
        row=dict(epoch=epoch,train_loss=total/count,learning_rate=lr,seconds=time.perf_counter()-epoch_tick,processed_from=resume_start,**measured)
        history.append(row);__import__('pandas').DataFrame(history).to_csv(root/'history.csv',index=False)
        print(kind,root.name,'epoch',epoch,'AUC',round(measured['roc_auc'],6),'seconds',round(row['seconds'],1),flush=True)
        epoch+=1;order=None;offset=0;save_progress();budget.beat()
        if cfg.get('patience') and epoch>cfg.get('min_epochs',0) and bad>=cfg['patience']:
            stop='EARLY_STOPPING';break
    elapsed+=time.perf_counter()-started
    if not (root/'history.csv').exists():__import__('pandas').DataFrame(history).to_csv(root/'history.csv',index=False)
    # Fixed-epoch S6E7 uses final epoch, never its best heldout score.
    complete=stop!='BUDGET_STOP'
    selected='last.pt' if kind=='s6e7' else 'best.pt'
    if not (root/selected).exists():
        save_progress();selected='last.pt'
    cp=torch.load(root/selected,map_location='cpu',weights_only=False);model.load_state_dict(cp['model'])
    p=predict(model,kind,valid_x,valid_c,cfg['inference_batch']);np.save(root/'p_valid.npy',p)
    # Reload a separately constructed model, not only the same Python object.
    clone=build(kind,meta,cfg['seed']).cuda();clone.load_state_dict(cp['model'])
    replay=predict(clone,kind,valid_x,valid_c,cfg['inference_batch']);np.testing.assert_allclose(replay,p,rtol=1e-5,atol=1e-6)
    result=dict(status='COMPLETE' if complete else 'PARTIAL_BUDGET',kind=kind,config=cfg,meta=meta,selected_checkpoint=selected,completed_epochs=len(history),stop_reason=stop,selected_epoch=len(history) if kind=='s6e7' else best_epoch,metrics=metrics(valid_y,p),seconds=elapsed,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,checkpoint_reload_max_abs_error=float(np.max(np.abs(replay-p))),checkpoint_reload_scope='Whole validation, identical batch layout, unchanged tolerance',files=seal(root,list(dict.fromkeys([selected,'last.pt','p_valid.npy','history.csv']))),code=code,training_code=cp.get('code',code),finalized_without_additional_fit=bool(finalize_existing),finished_utc=now())
    write(root/'receipt.json',result)
    del clone,model,opt,x,c,yt,mi;gc.collect();torch.cuda.empty_cache();return result

