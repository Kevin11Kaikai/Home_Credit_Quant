from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .common import array_hash, digest


@dataclass
class PositiveSigmoid:
    theta: float = 0.0
    intercept: float = 0.0
    status: str = 'NOT_FITTED'
    message: str = ''
    data_hash: str = ''
    n: int = 0
    minority_n: int = 0
    unstable: bool = False
    clip: float = 1e-7
    iterations: int = 0

    @property
    def slope(self):
        return float(np.exp(self.theta))

    def fit(self, p, y, cfg):
        p, y = np.asarray(p,dtype=np.float64),np.asarray(y,dtype=np.float64)
        self.clip = cfg['clip']
        self.n = len(y)
        self.data_hash = digest({'probabilities':array_hash(p),'labels':array_hash(y)})
        if not len(y) or not np.isfinite(p).all() or not np.isin(y,[0,1]).all():
            self.status,self.message = 'FAILED','Invalid or empty C inputs'
            return self
        self.minority_n = int(min(y.sum(),len(y)-y.sum()))
        self.unstable = self.minority_n < cfg['rare_class_warning']
        if self.minority_n == 0:
            self.status,self.message = 'FAILED_SINGLE_CLASS','C has one class; no calibration fitted'
            return self
        z = logit(np.clip(p,self.clip,1-self.clip))
        def objective(params):
            slope = np.exp(params[0])
            linear = slope*z+params[1]
            value = np.mean(np.logaddexp(0,linear)-y*linear)
            residual = expit(linear)-y
            grad = np.array([np.mean(residual*slope*z),np.mean(residual)])
            return value,grad
        result = minimize(objective,np.zeros(2),method='L-BFGS-B',jac=True,
                          bounds=[cfg['theta_bounds'],cfg['intercept_bounds']],
                          options={'maxiter':cfg['maxiter'],'ftol':1e-12,'gtol':1e-8})
        self.iterations = int(result.nit)
        self.message = str(result.message)
        if result.success and np.isfinite(result.x).all() and np.isfinite(result.fun):
            self.theta,self.intercept = map(float,result.x)
            self.status = 'FITTED'
        else:
            self.status = 'FAILED_OPTIMIZATION'
        return self

    def predict(self,p):
        p = np.asarray(p,dtype=np.float64)
        if self.status != 'FITTED':
            return p.copy()
        return expit(self.slope*logit(np.clip(p,self.clip,1-self.clip))+self.intercept)

    def to_dict(self):
        return asdict(self)


def numeric_ties(raw, calibrated, clip=1e-7):
    raw,calibrated = np.asarray(raw),np.asarray(calibrated)
    order = np.argsort(raw,kind='stable')
    return {'clipped_input_count':int(((raw<clip)|(raw>1-clip)).sum()),
            'saturated_output_count':int(((calibrated==0)|(calibrated==1)).sum()),
            'unique_raw':len(np.unique(raw)), 'unique_calibrated':len(np.unique(calibrated)),
            'nondecreasing':bool(np.all(np.diff(calibrated[order])>=0)),
            'selection_rule':'raw probability then stable row_id, reused exactly for raw/calibrated'}
