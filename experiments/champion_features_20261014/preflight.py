"""Bounded final pre-fit checks; A/B only, no model fitting."""
import subprocess
import sys
import time
import numpy as np
import pyarrow.parquet as pq
import psutil
from common import HERE,ORIGINAL,DERIVED,read,write,sha,protected_hashes,now
from training import VARIANTS,variant_dir,completed_input


def main():
    start=time.monotonic()
    protected=protected_hashes()
    mapping=pq.read_table(DERIVED/'mapping_AB.parquet').to_pandas()
    split={s:np.load(ORIGINAL/f'splits/{s}.npy') for s in ['A','B','C','D']}
    assert np.array_equal(mapping.row_id.to_numpy(),np.r_[split['A'],split['B']])
    assert not set(mapping.row_id)&set(np.r_[split['C'],split['D']])
    assert mapping.case_id.is_unique
    y=np.r_[np.load(DERIVED/'y_A.npy'),np.load(DERIVED/'y_B.npy')]
    assert len(y)==len(mapping) and set(np.unique(y))=={0,1}
    inputs={}
    for v in VARIANTS:
        folder=variant_dir(v)
        assert completed_input(folder)
        s=read(folder/'state.json')
        n=np.load(folder/'num.npy',mmap_mode='r')
        c=np.load(folder/'cat.npy',mmap_mode='r')
        assert len(n)==len(c)==len(mapping)
        assert n.shape[1]==len(s['numeric_names']) and c.shape[1]==len(s['categorical_names'])
        assert len(set(s['numeric_names']+s['categorical_names']))==n.shape[1]+c.shape[1]
        assert not set(s['numeric_names']+s['categorical_names'])&{'case_id','row_id','target','index_in_full','WEEK_NUM','date_decision'}
        for st in range(0,len(n),8192):
            assert np.isfinite(n[st:st+8192]).all()
        for j,card in enumerate(s['cardinalities']):
            assert c[:,j].min()>=0 and c[:,j].max()<card
        missing=sum(name.endswith('__missing') for name in s['numeric_names'])
        inputs[v]={'numeric_including_indicators':n.shape[1],'missing_indicators':missing,
                   'categorical':c.shape[1],'state_sha256':sha(folder/'state.json')}
        del n,c
    tests={}
    for executable,file in [(HERE/'.venv_fe/Scripts/python.exe','test_features.py'),(sys.executable,'test_training.py')]:
        r=subprocess.run([str(executable),'-m','pytest','-q',str(HERE/file)],capture_output=True,text=True)
        tests[file]={'returncode':r.returncode,'output':r.stdout+r.stderr}
        assert r.returncode==0,tests[file]['output']
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],capture_output=True,text=True)
    assert gpu.returncode==0
    import torch
    assert torch.cuda.is_available()
    write(HERE/'preflight.json',dict(status='PASS',utc=now(),seconds=time.monotonic()-start,
          original_protected_hashes=protected,inputs=inputs,tests=tests,hardware={'gpu':gpu.stdout.strip(),
          'RAM_total':psutil.virtual_memory().total,'RAM_available':psutil.virtual_memory().available},
          dates={s:dict(n=int((mapping.split==s).sum()),min=str(mapping.loc[mapping.split==s,'date_decision'].min()),
                         max=str(mapping.loc[mapping.split==s,'date_decision'].max())) for s in ['A','B']},
          no_C_D_labels_loaded=True,independent_reviewer=False))
    print('PREFLIGHT PASS',inputs,flush=True)


if __name__=='__main__':main()
