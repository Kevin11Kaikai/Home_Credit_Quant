"""Synthetic QA; does not access any real C/D labels or run a formal fit."""
import sys,types,copy,inspect
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd,torch
import pytest
import models
from common import sha,write,saved,metrics
from campaign import score_postprocess,champion_tree
from data import DenseAdapter

def test_champion_weights_postprocess():
    from common import read,HERE
    w=read(HERE/'config.json')['champion']['weights'];assert sum(w.values())==1
    p=np.array([.01,.1,.2,.3]);weeks=np.arange(4)
    np.testing.assert_allclose(score_postprocess(p,weeks),[0,.07,.2,.3]);assert p[0]==.01

def test_official_excel_sdpa_equivalence_and_gradients():
    from vendor.excel import MultiheadAttention
    torch.manual_seed(42)
    m=models.build('excel',{'n_num':6});attn=next(a for a in m.modules() if isinstance(a,MultiheadAttention))
    original=copy.deepcopy(attn);original.forward=types.MethodType(MultiheadAttention.forward,original)
    attn.eval();original.eval();x=torch.randn(3,6,128,requires_grad=True);xx=x.detach().clone().requires_grad_()
    a=attn(x,x,None,None);b=original(xx,xx,None,None)
    torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
    a.sum().backward();b.sum().backward();torch.testing.assert_close(x.grad,xx.grad,rtol=1e-4,atol=1e-6)

def test_accumulated_bce_gradient_matches_full_batch():
    torch.manual_seed(42);a=torch.nn.Linear(7,1);b=copy.deepcopy(a);x=torch.randn(13,7);y=torch.randint(0,2,(13,)).float()
    torch.nn.functional.binary_cross_entropy_with_logits(a(x).ravel(),y).backward()
    for ix in torch.arange(13).split(4):
        (torch.nn.functional.binary_cross_entropy_with_logits(b(x[ix]).ravel(),y[ix])*len(ix)/13).backward()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad)

def test_fold_target_encoding_excludes_heldout_labels():
    from catstat import TargetEncoder
    x=pd.DataFrame({'x':np.repeat(np.arange(40),5),'cat':np.tile(np.arange(5),40)})
    y=np.tile([0,1],100);train=np.arange(150);hold=np.arange(150,200)
    def fit(labels):
        e=TargetEncoder(cols=['x','cat'],target_type='binary',numeric='direct',cv=5,random_state=42)
        t=e.fit_transform(x.iloc[train],labels[train]);return np.asarray(t),np.asarray(e.transform(x.iloc[hold]))
    a,b=fit(y);yy=y.copy();yy[hold]=1-yy[hold];aa,bb=fit(yy)
    np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)

def test_columnwise_te_matches_joint_api():
    from catstat import TargetEncoder
    x=pd.DataFrame({'a':np.tile(np.arange(8),20),'b':np.repeat(np.arange(20),8)});y=np.tile([0,1,1,0],40)
    def fit(cols):return np.asarray(TargetEncoder(cols=cols,target_type='binary',numeric='direct',cv=5,random_state=42).fit_transform(x[cols],y))
    np.testing.assert_array_equal(fit(['a','b']),np.column_stack([fit(['a']),fit(['b'])]))

def test_inner_crossfit_smoothing_excludes_heldout_labels():
    from catstat import TargetEncoder
    from sklearn.model_selection import StratifiedKFold
    x=pd.DataFrame({'repeat':np.tile(np.arange(13),20),'unique':np.arange(260)})
    y=np.tile([0,1,0,0],65)
    splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y))
    held=splits[0][1]
    def encode(labels):
        return np.asarray(TargetEncoder(cols=list(x),target_type='binary',numeric='direct',smooth='auto',cv=splits,random_state=42).fit_transform(x,labels))
    a=encode(y);yy=y.copy();yy[held]=1-yy[held];b=encode(yy)
    np.testing.assert_array_equal(a[held],b[held])

def test_dense_train_only_unknown_missing_constant():
    a=np.array([[1,np.nan,2],[1,3,3],[1,5,2],[1,4,3]],float);s={'numeric':['a','b']}
    p=DenseAdapter().fit(a,s);v=np.array([[9,np.nan,99],[1,100,2]],float);x=p.transform(v);z=p.transform(v[:1]);assert np.isfinite(x).all();np.testing.assert_array_equal(x[:1],z);assert p.median[0]==1

def test_hash_tampering_detected(tmp_path):
    (tmp_path/'p.txt').write_text('before');write(tmp_path/'receipt.json',dict(files={'p.txt':sha(tmp_path/'p.txt')}));saved(tmp_path);(tmp_path/'p.txt').write_text('after')
    with pytest.raises(RuntimeError):saved(tmp_path)

def test_new_training_interface_has_no_cd():
    assert not set(inspect.signature(models.train_neural).parameters)&{'y_C','y_D','test_callback','test_y'}
    assert 'D' not in inspect.signature(champion_tree).parameters

def test_full_s6e7_count_and_averaging():
    from common import read,HERE
    c=read(HERE/'config.json')['s6e7'];assert c['folds']*len(c['member_seeds'])==28 and c['epochs']==16
    p=np.arange(28)/28;assert np.mean(p)==sum(p)/28

