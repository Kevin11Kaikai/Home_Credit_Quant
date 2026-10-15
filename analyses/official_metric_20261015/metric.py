"""Published Home Credit 2024 stability formula on original WEEK_NUM groups."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata


def terms(gini, independent=False):
    y = np.asarray(gini, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError('At least two finite weekly Gini values are required')
    x = np.arange(len(y), dtype=np.float64)
    if independent:
        xc, yc = x-x.mean(), y-y.mean()
        slope = float(np.dot(xc, yc)/np.dot(xc, xc))
        intercept = float(y.mean()-slope*x.mean())
        residual = y-(slope*x+intercept)
        residual_std = float(np.sqrt(np.mean((residual-residual.mean())**2)))
    else:
        slope, intercept = map(float, np.polyfit(x, y, 1))
        residual_std = float(np.std(y-(slope*x+intercept), ddof=0))
    mean = float(y.mean())
    falling = float(88.0*min(0.0, slope))
    variation = float(-0.5*residual_std)
    return dict(mean_weekly_gini=mean, slope=slope, intercept=intercept,
                falling_penalty=falling, residual_std=residual_std,
                variation_penalty=variation, stability_score=mean+falling+variation)


def rank_auc(y, p):
    """Mann-Whitney rank formula, including half credit for score ties."""
    y, p = np.asarray(y), np.asarray(p)
    n1 = int(y.sum()); n0 = len(y)-n1
    if n1 == 0 or n0 == 0:
        raise ValueError('Single-class week')
    return float((rankdata(p, method='average')[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def score(y, p, weeks):
    y, p, weeks = np.asarray(y), np.asarray(p, dtype=float), np.asarray(weeks)
    if y.ndim != 1 or p.shape != y.shape or weeks.shape != y.shape:
        raise ValueError('Mismatched input shapes')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError('Binary labels and finite scores required')
    unique = np.unique(weeks)
    if len(unique) < 2 or not np.all(np.diff(unique) == 1):
        raise ValueError('Require at least two contiguous original WEEK_NUM groups')
    rows = []
    for w in unique:
        mask = weeks == w; yy, pp = y[mask], p[mask]
        two = len(np.unique(yy)) == 2
        auc = float(roc_auc_score(yy, pp)) if two else None
        alt = rank_auc(yy, pp) if two else None
        rows.append(dict(WEEK_NUM=int(w), n=int(mask.sum()), events=int(yy.sum()),
                         event_rate=float(yy.mean()), single_class=not two,
                         roc_auc=auc, rank_auc=alt, gini=2*auc-1 if two else None))
    table = pd.DataFrame(rows)
    invalid = int(table.single_class.sum())
    common = dict(n=len(y), weeks=len(unique), first_week=int(unique[0]),
                  last_week=int(unique[-1]), single_class_weeks=invalid,
                  min_week_n=int(table.n.min()), min_week_events=int(table.events.min()))
    if invalid:
        return dict(status='UNDEFINED_SINGLE_CLASS_WEEK', **common,
                    **{k:None for k in terms([0,0])}), table
    main = terms(table.gini)
    other = terms(2*table.rank_auc-1, independent=True)
    error = max(abs(main[k]-other[k]) for k in main)
    if error > 1e-10:
        raise AssertionError('Independent calculation mismatch')
    return dict(status='COMPLETE_POSTHOC', **common, **main,
                recomputation_max_absolute_error=error), table
