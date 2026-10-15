"""Fresh persisted-preprocessor replay on feature-only B rows; no labels."""
import joblib,numpy as np,pandas as pd
from common import *
from data import raw

def main():
    root=HERE/'derived/s6e7_0';r=saved(root);assert r
    x,s=raw();indices=np.array([0,1,17,123,512,1023,40595]);a=np.asarray(x[s['n_A']+indices])
    encs=joblib.load(root/'target_encoders.joblib');pre=joblib.load(root/'preprocessor.joblib')
    encoded=[]
    for j,(name,enc) in enumerate(zip(s['names'],encs)):
        # Match the source notebook and cached adapter's explicit float32 TE
        # cast before the quantile transform; float64 is a different pipeline.
        encoded.append(np.asarray(enc.transform(pd.DataFrame({name:a[:,j]})),dtype=np.float32).reshape(-1))
    augmented=np.column_stack([a,*encoded]);names=s['names']+[f'te_{j}' for j in range(len(encoded))]
    num,cat=pre.transform(pd.DataFrame(augmented,columns=names))
    cached_n=np.load(root/'B_num.npy',mmap_mode='r')[indices];cached_c=np.load(root/'B_cat.npy',mmap_mode='r')[indices]
    np.testing.assert_array_equal(num,cached_n);np.testing.assert_array_equal(cat,cached_c)
    checks={}
    for part in ['train','valid','B']:
        n=np.load(root/f'{part}_num.npy',mmap_mode='r');c=np.load(root/f'{part}_cat.npy',mmap_mode='r')
        for off in range(0,len(n),8192):
            assert np.isfinite(n[off:off+8192]).all();cc=c[off:off+8192];assert (cc>=0).all() and (cc<np.array(r['cat_cardinalities'])).all()
        checks[part]=dict(numeric_shape=n.shape,categorical_shape=c.shape,finite=True,categorical_bounds_valid=True)
    write(HERE/'reports/s6e7_preprocessor_reload_verification.json',dict(status='PASS',fold=0,scope='Actual complete prepared outer fold; persisted TE + numeric scaler/category mapping on 7 feature-only B rows, and all tensor bounds.',labels_read=False,exact_numeric_replay=True,exact_categorical_replay=True,checks=checks,source_receipt_sha256=sha(root/'receipt.json'),verified_utc=now()))
    print('S6E7_PREPROCESSOR_FRESH_RELOAD_PASS',flush=True)
if __name__=='__main__':main()
