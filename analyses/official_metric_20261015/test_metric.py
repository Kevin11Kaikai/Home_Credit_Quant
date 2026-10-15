import numpy as np
import pytest
from metric import score, terms, rank_auc


def fixture():
    return np.tile([0, 0, 1, 1], 3), np.repeat([69,70,71], 4)


def test_constant_zero():
    y,w=fixture();s,_=score(y,np.ones(12)*.03,w)
    assert s['stability_score']==0


def test_perfect_one():
    y,w=fixture();s,_=score(y,y*.8+.1,w)
    assert s['stability_score']==pytest.approx(1,abs=1e-12)


def test_decline_not_clipped_to_zero():
    s=terms([1,.5,0])
    assert s['slope']==pytest.approx(-.5)
    assert s['stability_score']==pytest.approx(-43.5)


def test_increase_gets_no_bonus():
    s=terms([0,.5,1])
    assert s['falling_penalty']==0
    assert s['stability_score']==pytest.approx(.5)


def test_population_residual_std():
    s=terms([.5,1,.5])
    assert s['residual_std']==pytest.approx(np.sqrt(1/18))
    assert s['stability_score']==pytest.approx(2/3-.5*np.sqrt(1/18))


def test_unequal_week_sizes_not_weighted():
    y,w=fixture();p=np.array([.1,.2,.8,.9,.1,.4,.35,.8,.1,.8,.2,.7])
    s,_=score(y,p,w)
    ids=np.r_[np.arange(12),np.tile(np.arange(4),9)]
    t,_=score(y[ids],p[ids],w[ids])
    assert s['stability_score']==pytest.approx(t['stability_score'])


def test_monotone_and_permuted_rows():
    y,w=fixture();p=np.array([.1,.2,.8,.9,.1,.4,.35,.8,.1,.8,.2,.7])
    a,_=score(y,p,w);r=np.random.default_rng(42).permutation(12)
    b,_=score(y[r],(p[r]**2),w[r])
    assert a['stability_score']==pytest.approx(b['stability_score'])


def test_ties_half_credit():
    assert rank_auc([0,1],[.2,.2])==.5
    assert rank_auc([0,0,1,1],[.1,.4,.35,.8])==.75


def test_single_class_not_dropped():
    y,w=fixture();y[:4]=0;s,t=score(y,np.arange(12)/12,w)
    assert s['status']=='UNDEFINED_SINGLE_CLASS_WEEK'
    assert s['stability_score'] is None and len(t)==3


def test_gapped_weeks_rejected():
    y,w=fixture();w[w==71]=73
    with pytest.raises(ValueError):score(y,np.zeros(12),w)
