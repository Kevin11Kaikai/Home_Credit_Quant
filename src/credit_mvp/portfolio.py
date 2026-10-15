from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .evaluation import week_start


def select_mask(p_raw,row_id,dates,q):
    if not 0<q<=1:
        raise ValueError('q must be in (0,1]')
    weeks=week_start(dates)
    mask=np.zeros(len(p_raw),dtype=bool)
    audit=[]
    for week in np.unique(weeks):
        ids=np.flatnonzero(weeks==week)
        keep=int(np.floor(q*len(ids)))
        order=np.lexsort((np.asarray(row_id)[ids],np.asarray(p_raw)[ids]))
        mask[ids[order[:keep]]]=True
        audit.append({'week':str(week),'q':q,'available':len(ids),'selected':keep,
                      'rounding_loss':float(q*len(ids)-keep),'zero_selected':keep==0})
    return mask,audit


def random_mask(row_id,dates,q,seed):
    # Canonical row order makes the random control invariant to input permutation.
    weeks=week_start(dates)
    rng=np.random.default_rng(seed)
    mask=np.zeros(len(row_id),dtype=bool)
    for week in np.unique(weeks):
        ids=np.flatnonzero(weeks==week)
        ids=ids[np.argsort(np.asarray(row_id)[ids],kind='stable')]
        keep=int(np.floor(q*len(ids)))
        mask[rng.permutation(ids)[:keep]]=True
    return mask


def loss_metrics(p,y,mask,lgd=.4,ead=1.0):
    n=int(np.sum(mask))
    exposure=n*ead
    if n==0:
        return {'n':0,'exposure':0.0,'actual_retention':0.0,'mean_probability':None,'observed_event_rate':None,
                'EL_hat':0.0,'Loss_proxy':0.0,'normalized_EL':None,'normalized_observed_proxy':None,'error_per_exposure':None,'absolute_error_per_exposure':None}
    el=float(np.sum(np.asarray(p)[mask])*ead*lgd)
    observed=float(np.sum(np.asarray(y)[mask])*ead*lgd)
    return {'n':n,'exposure':float(exposure),'actual_retention':n/len(mask),
            'mean_probability':float(np.mean(np.asarray(p)[mask])), 'observed_event_rate':float(np.mean(np.asarray(y)[mask])),
            'EL_hat':el,'Loss_proxy':observed,'normalized_EL':el/exposure,'normalized_observed_proxy':observed/exposure,
            'error_per_exposure':(el-observed)/exposure,'absolute_error_per_exposure':abs(el-observed)/exposure}


def stress_probability(p,odds_multiplier):
    p=np.asarray(p,dtype=np.float64)
    # Exact endpoints preserved; log-odds shift interpreted as an assumption.
    with np.errstate(divide='ignore'):
        return expit(logit(p)+np.log(odds_multiplier))


def evaluate_portfolios(predictions,y,row_id,dates,constant_probability,cfg):
    portfolios,stress,rounding=[],[],[]
    reference_counts={}
    for model_key,pair in predictions.items():
        model,seed=model_key
        for q in cfg['q']:
            mask,audit=select_mask(pair['raw'],row_id,dates,q)
            for item in audit:
                rounding.append({'model':model,'seed':seed,**item})
            if q in reference_counts:
                assert int(mask.sum())==reference_counts[q]
            reference_counts[q]=int(mask.sum())
            for variant,p in pair.items():
                portfolios.append({'model':model,'seed':seed,'variant':variant,'q':q,
                                   **loss_metrics(p,y,mask,cfg['scenarios']['Base']['lgd'],cfg['ead'])})
                for scenario,parameters in cfg['scenarios'].items():
                    values=stress_probability(p,parameters['odds_multiplier'])
                    el=float(np.sum(values[mask])*cfg['ead']*parameters['lgd'])
                    stress.append({'model':model,'seed':seed,'variant':variant,'q':q,'scenario':scenario,
                                   **parameters,'n':int(mask.sum()),'exposure':float(mask.sum()*cfg['ead']),
                                   'stress_EL':el,'normalized_stress_EL':el/(mask.sum()*cfg['ead']) if mask.any() else None,
                                   'status':'ILLUSTRATIVE_ASSUMPTION_NO_STRESS_LABELS'})
    constant=np.full(len(y),constant_probability)
    portfolios.append({'model':'Full_pool','seed':None,'variant':'constant_A_probability','q':1.0,
                       **loss_metrics(constant,y,np.ones(len(y),dtype=bool),cfg['scenarios']['Base']['lgd'],cfg['ead'])})
    for q in cfg['q']:
        for seed in cfg['random_seeds']:
            mask=random_mask(row_id,dates,q,seed)
            assert int(mask.sum())==reference_counts.get(q,int(mask.sum()))
            portfolios.append({'model':'Random','seed':seed,'variant':'constant_A_probability','q':q,
                               **loss_metrics(constant,y,mask,cfg['scenarios']['Base']['lgd'],cfg['ead'])})
    table=pd.DataFrame(portfolios)
    controls=table[table.model=='Random'].groupby('q')[['n','exposure','observed_event_rate','normalized_observed_proxy']].agg(['mean','std']).reset_index()
    controls.columns=['_'.join(c).rstrip('_') for c in controls.columns]
    return table,pd.DataFrame(stress),pd.DataFrame(rounding),controls
