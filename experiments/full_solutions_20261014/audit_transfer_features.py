"""Covariate-only representation audit; cannot alter the frozen feature set."""
import numpy as np,pandas as pd
from common import *
from data import raw

def main():
    path=HERE/'reports/exact_value_coverage_AB.csv'
    if path.exists():print('REUSE_COVARIATE_AUDIT',flush=True);return
    x,s=raw();n=s['n_A'];rows=[]
    for j,col in enumerate(s['names']):
        a=np.asarray(x[:n,j]);b=np.asarray(x[n:,j]);aa=a[np.isfinite(a)];bb=b[np.isfinite(b)]
        unique,counts=np.unique(aa,return_counts=True)
        seen=np.isin(bb,unique)
        rows.append(dict(feature=col,type='categorical' if col in s['categorical'] else 'numeric',A_nonmissing=len(aa),A_missing_rate=float(1-len(aa)/n),A_unique=len(unique),A_singleton_row_fraction=float(np.sum(counts==1)/max(1,len(aa))),A_repeated_value_row_fraction=float(counts[counts>1].sum()/max(1,len(aa))),B_nonmissing=len(bb),B_exact_value_seen_in_A_fraction=float(seen.mean()) if len(bb) else None))
    frame=pd.DataFrame(rows);frame.to_csv(path,index=False)
    write(HERE/'reports/exact_value_audit.json',dict(status='COMPLETE_COVARIATE_ONLY',columns=len(rows),numeric_columns=int((frame.type=='numeric').sum()),numeric_median_seen_B_fraction=float(frame.loc[frame.type=='numeric','B_exact_value_seen_in_A_fraction'].median()),no_labels_read=True,no_feature_selection=True,source_hash=sha(HERE/'derived/raw_cat/receipt.json'),file_sha256=sha(path),finished_utc=now()))
    print('COVARIATE_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()
