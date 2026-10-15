import json
import inspect
import numpy as np
import pytest
from training import numeric_fit,numeric_transform,category_fit,category_transform,load_dev,fit


def test_numeric_preprocessor_fits_only_A_and_roundtrips():
    x=np.array([1.,3.,np.nan,9.])
    s=numeric_fit(x,3)
    assert s==numeric_fit(np.array([1.,3.,np.nan,-999.]),3)
    assert s['median']==2.
    out=numeric_transform(x,s)
    assert np.allclose(out[0][:3],[-np.sqrt(1.5),np.sqrt(1.5),0])
    assert np.allclose(out[1][:3],[-1/np.sqrt(2),-1/np.sqrt(2),np.sqrt(2)])
    for a,b in zip(out,numeric_transform(x,json.loads(json.dumps(s)))):
        assert np.array_equal(a,b)


def test_constant_missing_and_infinite_values():
    assert numeric_transform([np.nan,np.inf],numeric_fit([np.nan,np.inf],2))==[]
    state=numeric_fit([4.,4.,np.nan,999.],3)
    assert not state['keep_numeric'] and state['add_missing']
    out=numeric_transform([4.,np.nan,np.inf],state)
    assert len(out)==1 and np.isfinite(out[0]).all()


def test_unknown_categories_and_A_only_vocabulary():
    x=['b','a',None,'new']
    state=category_fit(x,3)
    assert state==category_fit(x[:3]+['different'],3)=={'a':2,'b':3}
    assert category_transform(x,state).tolist()==[3,2,1,0]
    assert category_transform(x,json.loads(json.dumps(state))).tolist()==[3,2,1,0]


def test_feature_fit_does_not_accept_labels_or_C_D():
    assert list(inspect.signature(numeric_fit).parameters)==['values','n_a']
    assert list(inspect.signature(category_fit).parameters)==['values','n_a']
    source=inspect.getsource(load_dev)
    assert "'y_A.npy'" in source and "'y_B.npy'" in source
    assert 'y_C' not in source and 'y_D' not in source
    assert list(inspect.signature(fit).parameters)==['variant','family','seconds']
