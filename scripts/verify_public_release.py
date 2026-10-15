"""Verify the published bytes and aggregate statuses without data or training."""
from pathlib import Path
import argparse, csv, hashlib, json, math, statistics

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def csv_rows(path):
    with (ROOT/path).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def verify_official_metric():
    """Recompute aggregate arithmetic only; no individual records are public."""
    folder=Path('analyses/official_metric_20261015')
    receipt=read(folder/'RESULT_RECEIPT.json')
    assert receipt['status']=='COMPLETE_POSTHOC_OFFICIAL_FORMULA'
    assert receipt['kaggle_submission'] is False and receipt['leaderboard_comparable'] is False
    expected_files={**receipt['outputs'],**receipt['identity']['code'],
                    'PROTOCOL.json':receipt['identity']['protocol_sha256']}
    for name,expected in expected_files.items():
        assert hashlib.sha256((ROOT/folder/name).read_bytes()).hexdigest()==expected, name

    def close(actual,expected):
        assert math.isclose(float(actual),float(expected),rel_tol=0,abs_tol=1e-10),(actual,expected)

    checked=0; all_scores={}
    for split,week_count,n_records in [('B',9,40596),('D',23,56001)]:
        scores=csv_rows(folder/f'official_metric_{split}.csv')
        weekly=csv_rows(folder/f'official_weeks_{split}.csv')
        assert len(scores)==13 and len(weekly)==13*week_count
        keys=set()
        for row in scores:
            key=(row['model'],row['seed'],row['variant'])
            assert key not in keys
            keys.add(key)
            part=sorted([w for w in weekly if (w['model'],w['seed'],w['variant'])==key],key=lambda w:int(w['WEEK_NUM']))
            assert len(part)==week_count
            weeks=[int(w['WEEK_NUM']) for w in part]
            assert weeks==list(range(int(row['first_week']),int(row['last_week'])+1))
            assert int(row['n'])==sum(int(w['n']) for w in part)==n_records
            assert int(row['weeks'])==week_count and int(row['single_class_weeks'])==0
            assert all(w['single_class'].lower()=='false' for w in part)
            y=[float(w['gini']) for w in part]
            for w in part:
                close(w['gini'],2*float(w['roc_auc'])-1)
                close(w['roc_auc'],w['rank_auc'])
            x_mean=(week_count-1)/2; y_mean=statistics.fmean(y)
            slope=sum((i-x_mean)*(v-y_mean) for i,v in enumerate(y))/sum((i-x_mean)**2 for i in range(week_count))
            intercept=y_mean-slope*x_mean
            residual_std=statistics.pstdev(v-(slope*i+intercept) for i,v in enumerate(y))
            terms={'mean_weekly_gini':y_mean,'slope':slope,'intercept':intercept,
                   'falling_penalty':88*min(0,slope),'residual_std':residual_std,
                   'variation_penalty':-0.5*residual_std}
            terms['stability_score']=y_mean+terms['falling_penalty']+terms['variation_penalty']
            for name,value in terms.items():close(row[name],value)
            all_scores[(split,*key)]=row
            checked+=1
    for model in ['LightGBM','TabM']:
        for seed in ['42','0','1']:
            close(all_scores[('D',model,seed,'raw')]['stability_score'],
                  all_scores[('D',model,seed,'calibrated')]['stability_score'])
    summaries=csv_rows(folder/'seed_summary_D.csv')
    assert len(summaries)==4
    for row in summaries:
        values=[float(all_scores[('D',row['model'],seed,row['variant'])]['stability_score']) for seed in ['42','0','1']]
        assert int(row['n_seeds'])==3
        close(row['stability_mean'],statistics.fmean(values))
        close(row['stability_sd'],statistics.stdev(values))
    return checked


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',action='store_true');args=p.parse_args()
    manifest=read('RELEASE_MANIFEST.json')
    for name,expected in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise SystemExit('Publication hash mismatch: '+name)
    status=read('STATUS.json')
    assert status['core']=='CORE_COMPLETE' and status['absbox']=='EXTENSION_COMPLETE'
    assert status['followup']=='PAUSED_BY_USER'
    assert status['s6e7_completed_members']==0 and status['automatic_resume_allowed'] is False
    with (ROOT/'evidence/core/metrics.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==13 and {r['n'] for r in rows}=={'56001'}
    for model in ['LightGBM','TabM']:
        pair={r['variant']:r for r in rows if r['model']==model and r['seed']=='42'}
        assert float(pair['raw']['roc_auc'])==float(pair['calibrated']['roc_auc'])
        assert float(pair['calibrated']['log_loss'])>float(pair['raw']['log_loss'])
    assert status['official_metric_analysis']['status']=='COMPLETE_POSTHOC_OFFICIAL_FORMULA'
    official_rows=verify_official_metric()
    print(f'PASS: {len(manifest["files"])} publication files verified; 13 historical core metric rows; paused follow-up preserved.')
    print(f'PASS: {official_rows} post-hoc official-formula scores recomputed from public weekly aggregates; all three fixed D seeds retained.')
    if args.summary:print(json.dumps(status,indent=2))
    print('This verifies published bytes and aggregate consistency, not new inference or independent replication.')

if __name__=='__main__':main()
