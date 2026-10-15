"""Direct NumPy arithmetic check of saved author aggregates; no targets."""
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np
import polars as pl
from common import HERE,RAW,DERIVED,write,now,sha


def main():
    mapping=pl.read_parquet(DERIVED/'mapping_AB.parquet')
    table=pl.read_parquet(DERIVED/'lgb/tables/tax_a.parquet').sort('case_id').head(20)
    ids=table['case_id']
    raw=pl.scan_parquet(RAW/'train_tax_registry_a_1.parquet').filter(pl.col('case_id').is_in(ids)).collect()
    checks=[]
    for variant in ['lgb','cat']:
        actual=pl.read_parquet(DERIVED/variant/'tables/tax_a.parquet').filter(pl.col('case_id').is_in(ids))
        n,error=0,0.
        for ident in ids:
            v=raw.filter(pl.col('case_id')==ident).sort('num_group1')['amount_4527230A'].to_numpy().astype(float)
            valid=v[np.isfinite(v)]
            expected={'max_amount_4527230A':np.max(valid) if len(valid) else np.nan,
                      'min_amount_4527230A':np.min(valid) if len(valid) else np.nan,
                      'mean_amount_4527230A':np.mean(valid) if len(valid) else np.nan,
                      'std_amount_4527230A':np.std(valid,ddof=1) if len(valid)>1 else np.nan}
            if variant=='lgb':
                vv=np.nan_to_num(v,nan=0.)
                expected['max-min_gap_depth2_amount_4527230A']=vv.max()-vv.min()
                expected['first_amount_4527230A']=v[0]
                expected['last_amount_4527230A']=v[-1]
            row=actual.filter(pl.col('case_id')==ident).row(0,named=True)
            for name,e in expected.items():
                a=row[name]
                a=np.nan if a is None else float(a)
                assert np.isclose(a,e,rtol=1e-10,atol=1e-9,equal_nan=True),(variant,name)
                if np.isfinite(a) and np.isfinite(e):error=max(error,abs(a-e))
                n+=1
        checks.append({'variant':variant,'cases':len(ids),'arithmetic_comparisons':n,'max_abs_error':error})
    write(HERE/'aggregate_arithmetic_check.json',dict(status='PASS',utc=now(),checks=checks,
          source_sha256=sha(RAW/'train_tax_registry_a_1.parquet'),targets_loaded=False,
          scope='One real tax table, 20 cases, separate NumPy calculation; not an independent reviewer or exhaustive validation'))
    print(checks)


if __name__=='__main__':main()
