from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .models import check_probabilities


def metrics(y,p):
    y,p=np.asarray(y),np.asarray(p,dtype=np.float64)
    check_probabilities(p)
    two=len(np.unique(y))==2
    return {'roc_auc':float(roc_auc_score(y,p)) if two else None,
            'average_precision':float(average_precision_score(y,p)) if two else None,
            'log_loss':float(log_loss(y,p,labels=[0,1])),
            'brier':float(brier_score_loss(y,p)),
            'mean_probability':float(np.mean(p)),
            'observed_event_rate':float(np.mean(y)),
            'probability_bias':float(np.mean(p)-np.mean(y)), 'n':len(y)}


def week_start(dates):
    dates=pd.DatetimeIndex(dates)
    return (dates.normalize()-pd.to_timedelta(dates.dayofweek,unit='D')).to_numpy(dtype='datetime64[D]')


def reliability_edges(p,n_bins=10):
    p=np.asarray(p,dtype=np.float64)
    check_probabilities(p)
    return np.unique(np.r_[0.0,np.quantile(p,np.linspace(0,1,n_bins+1))[1:-1],1.0])


def reliability(y,p,edges):
    bins=np.clip(np.searchsorted(edges,p,side='right')-1,0,len(edges)-2)
    rows=[]
    for i in range(len(edges)-1):
        mask=bins==i
        rows.append({'bin':i,'lower':float(edges[i]),'upper':float(edges[i+1]),'n':int(mask.sum()),
                     'mean_probability':float(np.mean(p[mask])) if mask.any() else None,
                     'observed_event_rate':float(np.mean(y[mask])) if mask.any() else None})
    return rows


def weekly(y,p,dates,small_n):
    weeks=week_start(dates)
    result=[]
    for week in np.unique(weeks):
        mask=weeks==week
        m=metrics(y[mask],p[mask])
        result.append({'week':str(week),**m,'small_sample':int(mask.sum())<small_n,'single_class':len(np.unique(y[mask]))<2})
    return result


def moving_block_bootstrap(y,dates,p_lgb,p_tabm,cfg):
    weeks=week_start(dates)
    distinct=np.unique(weeks)
    if len(distinct)<cfg['bootstrap_min_weeks']:
        return {'status':'NOT_EXECUTED_INSUFFICIENT_WEEKS','weeks':len(distinct),'min_weeks':cfg['bootstrap_min_weeks']},pd.DataFrame()
    # Require truly adjacent calendar weeks; do not call separated weeks a 2-week block.
    if np.any(np.diff(distinct)!=np.timedelta64(7,'D')):
        return {'status':'NOT_EXECUTED_GAPPED_CALENDAR','weeks':len(distinct)},pd.DataFrame()
    length=cfg['bootstrap_block_weeks']
    row_groups=[np.flatnonzero(weeks==week) for week in distinct]
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    records=[]
    for repeat in range(cfg['bootstrap_replicates']):
        starts=rng.integers(0,len(distinct)-length+1,size=int(np.ceil(len(distinct)/length)))
        picked=np.concatenate([np.arange(s,s+length) for s in starts])[:len(distinct)]
        ids=np.concatenate([row_groups[w] for w in picked])
        # Both models share exactly the same repeated rows in each paired draw.
        for variant in ['raw','calibrated']:
            left=metrics(y[ids],p_lgb[variant][ids])
            right=metrics(y[ids],p_tabm[variant][ids])
            records.append({'replicate':repeat,'variant':variant,'auc_diff':None if left['roc_auc'] is None else right['roc_auc']-left['roc_auc'],
                            'log_loss_diff':right['log_loss']-left['log_loss'], 'blocks':','.join(map(str,starts))})
    table=pd.DataFrame(records)
    summary={'status':'COMPLETE','direction':'TabM minus LightGBM','replicates':cfg['bootstrap_replicates'],'block_weeks':length,
             'scope':'Fixed fitted models; short-range temporal sampling uncertainty. Missing customer IDs and longer-range dependence remain limitations.', 'intervals':{}}
    for variant in ['raw','calibrated']:
        summary['intervals'][variant]={}
        for metric in ['auc_diff','log_loss_diff']:
            values=table.loc[table.variant==variant,metric].dropna().to_numpy()
            summary['intervals'][variant][metric]={'valid_replicates':len(values),'lower':float(np.quantile(values,.025)) if len(values) else None,'upper':float(np.quantile(values,.975)) if len(values) else None}
    return summary,table
