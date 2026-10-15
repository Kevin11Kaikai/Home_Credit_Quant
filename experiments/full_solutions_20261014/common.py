from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
from pathlib import Path
import hashlib,json,time,random
from datetime import datetime,timezone
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
PREVIOUS=HERE.parent/'champion_features_20261014'
ORIGINAL=Path(os.environ.get('CREDIT_MVP_CORE_RUN', PROJECT/'runs/homecredit_temporal_v1_20261014')).resolve()

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');t.replace(p)
def seal(folder,names):return {n:sha(folder/n) for n in names}
def verify(folder,hashes):
    for n,h in hashes.items():
        if sha(folder/n)!=h:raise RuntimeError('Artifact changed: '+str(folder/n))
def saved(folder):
    p=folder/'receipt.json'
    if not p.exists():return None
    r=read(p);verify(folder,r['files']);return r
def metrics(y,p):
    from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
    p=np.asarray(p,dtype=float)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return dict(roc_auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),brier=float(brier_score_loss(y,p)),mean_probability=float(p.mean()),observed_event_rate=float(y.mean()),n=len(y))
def protected():
    f=read(ORIGINAL/'FROZEN_MANIFEST.json')
    for p,h in f['files'].items():assert sha(p)==h,p
    for p,h in f['code'].items():assert sha(PROJECT/p)==h,p
    paths=[ORIGINAL/n for n in ['FROZEN_MANIFEST.json','TEST_EVALUATION.json','metrics.csv','predictions_D.parquet','budget.json']]
    paths += [PREVIOUS/n for n in ['PROTOCOL_LOCK.json','acceptance.json','budget.json'] if (PREVIOUS/n).exists()]
    return {str(p):sha(p) for p in paths}
def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
def rng_state():
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
    import torch
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])
def checkpoint(p,obj):
    import torch
    p=Path(p);t=p.with_suffix('.tmp');torch.save(obj,t);t.replace(p)

class Budget:
    """Persistent charged wall time; interruption never resets elapsed work."""
    def __init__(self):
        self.path=HERE/'budget.json'
        if not self.path.exists():write(self.path,dict(used={},active=None))
        self.state=read(self.path)
    def start(self,name):
        if self.state['active']:
            import psutil
            if self.state['active']['pid']!=os.getpid() and psutil.pid_exists(self.state['active']['pid']):
                raise RuntimeError('A campaign process already owns this budget')
            # A prior interrupted process may have run since its last heartbeat.
            a=self.state['active'];self.state['used'][a['name']]=self.state['used'].get(a['name'],0)+max(0,time.time()-a['heartbeat'])
        self.state['active']=dict(name=name,heartbeat=time.time(),pid=os.getpid());write(self.path,self.state)
    def beat(self):
        a=self.state['active'];tick=time.time()
        self.state['used'][a['name']]=self.state['used'].get(a['name'],0)+max(0,tick-a['heartbeat']);a['heartbeat']=tick;write(self.path,self.state)
    def stop(self):
        self.beat();self.state['active']=None;write(self.path,self.state)
    def used(self,prefix):return sum(v for k,v in self.state['used'].items() if k.startswith(prefix))

