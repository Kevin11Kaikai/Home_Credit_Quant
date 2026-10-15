"""Cached-result arithmetic and idempotent-run verification; no new fits."""
import subprocess
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
from common import HERE,PROJECT,DERIVED,ORIGINAL,read,write,sha,now,protected_hashes
from training import protocol


def main():
    p=protocol()
    table=pd.read_csv(HERE/'development_metrics_B.csv')
    y=np.load(DERIVED/'y_B.npy')
    checked={};files={}
    for row in table.itertuples():
        if row.variant=='original':
            pred=HERE/'baseline_replay'/(row.model+'.npy')
            ev=read(HERE/'baseline_replay'/(row.model+'.json'))
        else:
            folder=HERE/'fits'/row.variant/row.model
            pred=folder/'predictions_B.npy';ev=read(folder/'evaluation.json')
            result=read(folder/'result.json')
            assert sha(result['checkpoint'])==result['checkpoint_sha256']==ev['checkpoint_sha256']
            files[str(folder.relative_to(HERE)/'result.json')]=sha(folder/'result.json')
            files[str(folder.relative_to(HERE)/'evaluation.json')]=sha(folder/'evaluation.json')
            assert abs(row.roc_auc-result['B']['roc_auc'])<1e-6
            assert abs(row.log_loss-result['B']['log_loss'])<1e-6
        assert sha(pred)==ev['prediction_sha256']
        probs=np.load(pred)
        assert len(probs)==len(y) and np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
        recalculated={'roc_auc':roc_auc_score(y,probs),'average_precision':average_precision_score(y,probs),
                      'log_loss':log_loss(y,probs,labels=[0,1]),'brier':brier_score_loss(y,probs)}
        for name,value in recalculated.items():assert abs(value-getattr(row,name))<1e-12
        checked[row.variant+'/'+row.model]={'n':len(probs),'status':'MATCH_CACHED_ARITHMETIC'}
        files[str(pred.relative_to(HERE))]=sha(pred)
    before=sha(HERE/'budget.json')
    csv_hash=sha(HERE/'development_metrics_B.csv')
    result=subprocess.run([sys.executable,str(HERE/'training.py'),'run'],cwd=PROJECT,capture_output=True,text=True)
    (HERE/'logs/idempotent_run.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    assert result.returncode==0
    assert sha(HERE/'budget.json')==before,'Cached run changed budget'
    assert sha(HERE/'development_metrics_B.csv')==csv_hash,'Cached run changed metrics'
    assert 'START ' not in result.stdout,'Cached run started a fit'
    for name,expected in files.items():assert sha(HERE/name)==expected
    assert protected_hashes()==p['original_protected_hashes']
    write(HERE/'verification.json',dict(status='PASS',utc=now(),metric_rows_checked=checked,
          cached_run_exit_code=result.returncode,no_new_fits_on_cached_run=True,
          budget_unchanged=True,original_artifacts_unchanged=True,
          probability_files_finite_and_in_range=True,metric_tolerance=1e-12,
          independent_reviewer=False,no_C_D_labels_loaded=True,
          preflight_sha256=sha(HERE/'preflight.json'),aggregate_check_sha256=sha(HERE/'aggregate_arithmetic_check.json'),
          verified_files=files))
    print('FINAL_VERIFICATION_PASS',len(checked),'metric rows; cached run creates no fits')


if __name__=='__main__':main()
