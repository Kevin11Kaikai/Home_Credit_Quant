from datetime import date
import numpy as np
import polars as pl
import pytest
from features import aggregate, fit_filter, load_author, join_applications, EXCLUDED


@pytest.mark.parametrize('variant', ['lgb', 'cat'])
def test_author_numeric_fixture(variant):
    frame = pl.DataFrame({'case_id':[3,3,3,4], 'num_group1':[2,0,1,0],
                          'amount_1A':[9.,1.,5.,None]})
    out = aggregate(frame, variant, 'train_applprev_1_0.parquet', 1).sort('case_id')
    assert out['max_amount_1A'][0] == 9
    assert out['min_amount_1A'][0] == 1
    assert out['mean_amount_1A'][0] == 5
    assert out['std_amount_1A'][0] == 4
    assert out['max_amount_1A'][1] is None


def test_category_first_last_follow_group_order():
    f = pl.DataFrame({'case_id':[1,1,1], 'num_group1':[2,0,1], 'label_1M':['z','a','b']})
    out = aggregate(f,'cat','train_applprev_1_0.parquet',1)
    assert out['first_label_1M'][0]=='a'
    assert out['last_label_1M'][0]=='z'
    assert out['n_unique_label_1M'][0]==3


def test_train_only_filter():
    a = pl.Series([True,True,False,False])
    s1 = pl.Series('x_1A',[1.,2.,None,None])
    s2 = pl.Series('x_1A',[1.,2.,999.,-999.])
    assert fit_filter(s1,a,'cat') == fit_filter(s2,a,'cat') == None
    cat = pl.Series('x_1M',['a','a','b','c'])
    assert fit_filter(cat,a,'cat') == 'A_string_cardinality_not_in_2_to_50'


def test_metadata_and_target_are_not_features():
    assert {'target','case_id','WEEK_NUM','date_decision','index_in_full'} <= EXCLUDED
    with pytest.raises(AssertionError):
        aggregate(pl.DataFrame({'case_id':[1],'target':[1]}),'cat','train_base.parquet',0)


def test_date_delta_is_relative_to_application():
    ns = load_author('cat')
    f = pl.DataFrame({'case_id':[1], 'date_decision':[date(2020,1,5)],
                      'MONTH':[202001], 'latest_1D':[date(2020,1,2)]})
    out = ns['Pipeline'].handle_dates(f)
    assert out['latest_1D'][0] == -3
    assert 'date_decision' not in out.columns


def test_only_named_feature_definitions_are_loaded():
    ns = load_author('cat')
    assert 'models' not in ns and 'CatBoostClassifier' not in ns and 'subprocess' not in ns
    assert set(ns)-{'__builtins__'} == {'pl','np','Pipeline','Aggregator'}


def test_join_preserves_order_with_chunks_missing_cases_and_int_widths():
    base = pl.concat([pl.DataFrame({'case_id':[7,2]}),pl.DataFrame({'case_id':[8,1]})],rechunk=False)
    src = pl.DataFrame({'case_id':pl.Series([1,7],dtype=pl.Int32),'value':[10.,70.]})
    out = join_applications(base,src)
    assert out['case_id'].to_list() == [7,2,8,1]
    assert out['value'].to_list() == [70.,None,None,10.]


def test_empty_source_preserves_applications():
    base=pl.DataFrame({'case_id':[1,2]})
    out=join_applications(base,pl.DataFrame({'case_id':pl.Series([],dtype=pl.Int32)}))
    assert out.equals(base)
