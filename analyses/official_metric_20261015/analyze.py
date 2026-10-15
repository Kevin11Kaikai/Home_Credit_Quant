"""Cache-only official-formula analysis. Does not import or fit any model."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from sklearn.metrics import roc_auc_score, average_precision_score
from metric import score

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False)+'\n',encoding='utf-8')


def md_table(frame, columns):
    out=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,r in frame.iterrows():
        out.append('| '+' | '.join(f'{r[c]:.6f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in columns)+' |')
    return '\n'.join(out)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=HERE.parents[1])
    args=parser.parse_args()
    project=args.project.resolve();run=project/'runs/homecredit_temporal_v1_20261014'
    fe=project/'experiments/champion_features_20261014';full=project/'experiments/full_solutions_20261014'
    tick=time.perf_counter();pa.set_cpu_count(4)
    protocol=read(HERE/'PROTOCOL.json')
    te=read(run/'TEST_EVALUATION.json');frozen=read(run/'FROZEN_MANIFEST.json')
    mapping_receipt=read(fe/'derived/mapping_receipt.json')
    data_audit=read(run/'data_audit.json')
    b_receipt=read(full/'reports/report_receipt.json')
    core_metrics=pd.read_csv(run/'metrics.csv')
    b_metrics=pd.read_csv(full/'reports/comparison_B.csv')
    assert len(b_metrics)==13 and len(core_metrics)==13
    inputs={}
    def verify(path,expected=None):
        path=Path(path);actual=sha(path)
        if expected is not None:assert actual==expected,'Input hash mismatch: '+str(path)
        inputs[str(path.relative_to(project))]=actual
    for name,expected in frozen['code'].items():verify(project/name,expected)
    for name,expected in frozen['files'].items():verify(Path(name),expected)
    verify(run/'FROZEN_MANIFEST.json',te['manifest_sha256'])
    for name in ['TEST_EVALUATION.json','PROTOCOL_LOCK.json','budget.json','events.jsonl']:verify(run/name)
    for name in ['predictions_D.parquet','metrics.csv','matched_seeds.json']:
        verify(run/name,te['outputs'][name])
    for name in ['x_meta.npy','y.npy']:
        p=project/'data/homecredit-default'/name;verify(p,data_audit['files'][str(p)])
    verify(fe/'raw/train_base.parquet',mapping_receipt['raw_base_sha256'])
    verify(fe/'derived/mapping_AB.parquet',mapping_receipt['mapping_sha256'])
    for name in ['y_B.npy','mapping_receipt.json']:verify(fe/'derived'/name)
    for path,expected in b_receipt['source_prediction_hashes'].items():verify(Path(path),expected)
    verify(full/'reports/comparison_B.csv',b_receipt['files']['comparison_B.csv'])
    for folder in [fe,full]:
        for name in ['PROTOCOL_LOCK.json','budget.json','acceptance.json']:
            if (folder/name).exists():verify(folder/name)
    pause=read(full/'PAUSE_RECEIPT.json')
    for name in ['PAUSE_RECEIPT.json','PAUSE_REQUESTED.json','campaign_state.json']:verify(full/name)
    verify(full/'runs/excel/last.pt',pause['resume_point']['checkpoint_sha256'])
    assert read(full/'campaign_state.json')['status']=='PAUSED_BY_USER'
    assert read(full/'budget.json')['active'] is None
    assert not list((full/'runs').glob('s6e7*/receipt.json'))
    code={p.name:sha(p) for p in HERE.glob('*.py')}
    identity=dict(protocol_sha256=sha(HERE/'PROTOCOL.json'),code=code,inputs=inputs)
    if (HERE/'RESULT_RECEIPT.json').exists():
        old=read(HERE/'RESULT_RECEIPT.json')
        assert old['identity']==identity,'Completed analysis inputs/code changed; do not silently overwrite'
        for name,expected in old['outputs'].items():assert sha(HERE/name)==expected,name
        print('CACHE_ONLY: completed official-formula analysis verified.');return
    attempts=HERE/'attempts';attempts.mkdir(exist_ok=True)
    write(attempts/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),dict(
        created_utc=datetime.now(timezone.utc).isoformat(),identity=identity,
        access_scope='User-requested post-hoc B/D cached-prediction rescore; no C labels or model fitting'))

    # WEEK_NUM is not derived from a guessed calendar boundary.
    base=pd.read_parquet(fe/'raw/train_base.parquet',columns=['case_id','date_decision','WEEK_NUM'])
    meta=np.load(project/'data/homecredit-default/x_meta.npy',mmap_mode='r')
    target=np.load(project/'data/homecredit-default/y.npy',mmap_mode='r')
    mappings={};audits={}
    for split in ['B','D']:
        ids=np.load(run/f'splits/{split}.npy')
        source_positions=meta[ids,1].astype(np.int64)
        mapped=base.iloc[source_positions].copy().reset_index(drop=True)
        mapped['row_id']=ids
        mapped['date_decision']=pd.to_datetime(mapped.date_decision)
        assert np.array_equal(mapped.date_decision.to_numpy(dtype='datetime64[D]').astype(int),meta[ids,0])
        assert mapped.case_id.is_unique and mapped.row_id.is_unique
        # Filter to B or D before bringing raw target values into the result table.
        selected=ds.dataset(fe/'raw/train_base.parquet',format='parquet').to_table(
            columns=['case_id','target'],filter=ds.field('case_id').isin(mapped.case_id.to_numpy())).to_pandas()
        mapped=mapped.merge(selected,on='case_id',how='left',validate='one_to_one',sort=False)
        assert np.array_equal(mapped.row_id,ids)
        assert np.array_equal(mapped.target.to_numpy(),target[ids])
        if split=='B':
            old=pd.read_parquet(fe/'derived/mapping_AB.parquet')
            old=old[old.split=='B'].reset_index(drop=True)
            for col in ['row_id','case_id','WEEK_NUM']:assert np.array_equal(old[col],mapped[col]),col
            assert np.array_equal(np.load(fe/'derived/y_B.npy'),mapped.target)
        monday=mapped.date_decision-pd.to_timedelta(mapped.date_decision.dt.dayofweek,unit='D')
        mapped['calendar_monday']=monday
        groups=mapped.groupby('WEEK_NUM',sort=True)
        week_counts=groups.agg(n=('row_id','size'),events=('target','sum'),first_date=('date_decision','min'),last_date=('date_decision','max'))
        audits[split]=dict(n=len(mapped),official_weeks=len(week_counts),first_week=int(week_counts.index.min()),
            last_week=int(week_counts.index.max()),start=str(mapped.date_decision.min().date()),end=str(mapped.date_decision.max().date()),
            exact_date_match=True,exact_label_match=True,unique_application_ids=True,customer_independence_verified=False,
            monday_groups_per_official_week_max=int(groups.calendar_monday.nunique().max()),
            official_groups_per_monday_max=int(mapped.groupby('calendar_monday').WEEK_NUM.nunique().max()),
            min_week_n=int(week_counts.n.min()),min_week_events=int(week_counts.events.min()),
            contiguous_weeks=bool(np.all(np.diff(week_counts.index)==1)),
            contains_partial_boundary_weeks=True,row_level_mapping_exported=False)
        assert audits[split]['contiguous_weeks']
        mappings[split]=mapped

    detailed={};summaries={};max_errors=[];existing_metric_checks=0
    def evaluate(split,model,seed,variant,p,reference,label):
        nonlocal existing_metric_checks
        mapping=mappings[split];y=mapping.target.to_numpy();p=np.asarray(p)
        assert p.shape==y.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        pooled_auc=float(roc_auc_score(y,p));ap=float(average_precision_score(y,p))
        for k,v in [('roc_auc',pooled_auc),('average_precision',ap)]:
            assert abs(float(reference[k])-v)<1e-12,(model,k,v,reference[k])
            existing_metric_checks+=1
        result,weekly=score(y,p,mapping.WEEK_NUM)
        tag=dict(split=split,model=model,seed=int(seed),variant=variant,label=label)
        result.update(tag,pooled_roc_auc=pooled_auc,average_precision=ap,
                      observed_event_rate=float(y.mean()),evaluation_scope='posthoc_cached_D' if split=='D' else 'reused_development_B')
        for k,v in tag.items():weekly[k]=v
        dates=mapping.groupby('WEEK_NUM').date_decision.agg(['min','max'])
        weekly['first_date']=weekly.WEEK_NUM.map(dates['min']).dt.strftime('%Y-%m-%d')
        weekly['last_date']=weekly.WEEK_NUM.map(dates['max']).dt.strftime('%Y-%m-%d')
        summaries.setdefault(split,[]).append(result);detailed.setdefault(split,[]).append(weekly)
        if 'recomputation_max_absolute_error' in result:max_errors.append(result['recomputation_max_absolute_error'])

    pred=pd.read_parquet(run/'predictions_D.parquet')
    ids=mappings['D'].row_id.to_numpy()
    for _,r in core_metrics.iterrows():
        part=pred[(pred.model==r['model'])&(pred.seed==r['seed'])]
        assert part.row_id.is_unique and len(part)==len(ids)
        part=part.set_index('row_id').loc[ids]
        assert np.array_equal(pd.to_datetime(part.time).to_numpy(dtype='datetime64[D]'),mappings['D'].date_decision.to_numpy(dtype='datetime64[D]'))
        evaluate('D',r['model'],r['seed'],r['variant'],part['p_raw' if r['variant']=='raw' else 'p_cal'],r,r['model'])
    champion_files={'Champion reconstruction lgb':'lgb_B.npy','Champion reconstruction dense':'dense_B.npy',
                    'Champion reconstruction cat':'cat_B.npy','Champion reconstruction blend':'ensemble_B.npy',
                    'Champion reconstruction + score hack (not PD)':'score_hack_B.npy'}
    for _,r in b_metrics.iterrows():
        if r['name'] in champion_files:
            path=full/'results/champion'/champion_files[r['name']]
            variant='fixed_score_hack' if 'score hack' in r['name'] else 'raw'
        else:
            path=fe/('baseline_replay/'+r['model']+'.npy' if r['variant']=='original' else 'fits/'+r['variant']+'/'+r['model']+'/predictions_B.npy')
            variant='raw'
        evaluate('B',r['name'],42,variant,np.load(path),r,r['name'])

    frames={split:pd.DataFrame(rows) for split,rows in summaries.items()}
    weeks={split:pd.concat(rows,ignore_index=True) for split,rows in detailed.items()}
    delta=[]
    d=frames['D']
    for model in ['LightGBM','TabM']:
        for seed in [42,0,1]:
            pair=d[(d.model==model)&(d.seed==seed)].set_index('variant')
            w=weeks['D'];raw=w[(w.model==model)&(w.seed==seed)&(w.variant=='raw')].sort_values('WEEK_NUM')
            cal=w[(w.model==model)&(w.seed==seed)&(w.variant=='calibrated')].sort_values('WEEK_NUM')
            change=float(np.nanmax(np.abs(raw.gini.to_numpy()-cal.gini.to_numpy())))
            assert change<1e-12
            delta.append(dict(model=model,seed=seed,max_weekly_gini_difference=change,
                              calibrated_minus_raw_score=float(pair.loc['calibrated','stability_score']-pair.loc['raw','stability_score'])))
    seed_summary=d[d.model!='Constant'].groupby(['model','variant'],sort=False).agg(
        n_seeds=('seed','nunique'),stability_mean=('stability_score','mean'),stability_sd=('stability_score','std'),
        mean_gini=('mean_weekly_gini','mean'),mean_falling_penalty=('falling_penalty','mean'),
        mean_variation_penalty=('variation_penalty','mean')).reset_index()
    assert (seed_summary.n_seeds==3).all()
    # Recheck all protected files before saving the new analysis, including the paused checkpoint.
    for name,expected in inputs.items():assert sha(project/name)==expected,'Protected input changed: '+name
    verification=dict(status='PASS',existing_auc_ap_comparisons=existing_metric_checks,
        alternate_weekly_rank_auc_and_closed_form_ols=True,max_absolute_error=max(max_errors),
        independent_external_audit=False,raw_calibrated=delta,protected_files_unchanged=len(inputs),
        original_test_records=te['official_test_evaluation_records'],new_formal_test_record=False,
        new_inference=False,new_training=False,new_C_access=False,training_still_paused=True)
    write(HERE/'week_mapping_audit.json',audits);write(HERE/'verification.json',verification)
    for split in ['D','B']:
        frames[split].to_csv(HERE/f'official_metric_{split}.csv',index=False)
        weeks[split].to_csv(HERE/f'official_weeks_{split}.csv',index=False)
    seed_summary.to_csv(HERE/'seed_summary_D.csv',index=False)
    render(frames,weeks,seed_summary,audits)
    outputs={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in {'.csv','.png','.svg','.md','.json'} and p.name not in {'RESULT_RECEIPT.json','PROTOCOL.json'}}
    write(HERE/'RESULT_RECEIPT.json',dict(status='COMPLETE_POSTHOC_OFFICIAL_FORMULA',created_utc=datetime.now(timezone.utc).isoformat(),
        identity=identity,outputs=outputs,elapsed_seconds=time.perf_counter()-tick,
        scope=protocol['scope'],kaggle_submission=False,leaderboard_comparable=False))
    print(frames['D'][['model','seed','variant','stability_score','mean_weekly_gini','falling_penalty','variation_penalty']].to_string(index=False))
    print(frames['B'][['model','stability_score','mean_weekly_gini','falling_penalty','variation_penalty']].to_string(index=False))
    print(seed_summary.to_string(index=False))
    print('Verification:',json.dumps(verification))


def render(frames,weeks,seed_summary,audits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=frames['D'];main=d[d.seed==42]
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    colors={'LightGBM':'#167d8d','TabM':'#cf6726'}
    for model in ['LightGBM','TabM']:
        w=weeks['D'];w=w[(w.model==model)&(w.seed==42)&(w.variant=='raw')].sort_values('WEEK_NUM')
        r=main[(main.model==model)&(main.variant=='raw')].iloc[0]
        axes[0].plot(w.WEEK_NUM,w.gini,marker='o',ms=3,lw=1.4,label=model,color=colors[model])
        axes[0].plot(w.WEEK_NUM,r.intercept+r.slope*np.arange(len(w)),ls='--',lw=1,color=colors[model])
    axes[0].set(title='D: original WEEK_NUM; seed 42',xlabel='Official WEEK_NUM',ylabel='Weekly Gini = 2 AUC - 1')
    axes[0].grid(alpha=.2);axes[0].legend()
    subset=main[(main.model!='Constant')&(main.variant=='raw')]
    x=np.arange(len(subset));width=.25
    for j,(col,label,color) in enumerate([('mean_weekly_gini','Mean weekly Gini','#8293a5'),('falling_penalty','Trend penalty','#c85b5b'),('stability_score','Stability score','#167d8d')]):
        vals=subset[col].to_numpy();bars=axes[1].bar(x+(j-1)*width,vals,width,label=label,color=color)
        axes[1].bar_label(bars,fmt='%.4f',fontsize=9,padding=3)
    axes[1].set_xticks(x,subset.model);axes[1].axhline(0,color='#888',lw=.7)
    axes[1].set(title='Official formula on local D',ylabel='Score contribution')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3,fontsize=8);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Post-hoc cached-prediction analysis — not a Kaggle leaderboard score',fontsize=12)
    fig.savefig(HERE/'official_metric_D.png',dpi=170);fig.savefig(HERE/'official_metric_D.svg');plt.close(fig)
    cols=['model','variant','stability_score','mean_weekly_gini','falling_penalty','variation_penalty']
    text='''# 按 Home Credit 2024 官方公式的事后评价

这是用户于 2026-10-15 明确要求的固定缓存预测分析。没有训练、重新推理、改选模型/seed、读取 C 标签或提交 Kaggle；原冻结测试评价及暂停状态保持不变。本指标是在看过原始结果后新增的，不能伪装为原先预注册主指标。

## 指标和分组

公式：`mean(weekly_gini) + 88 × min(0, slope) − 0.5 × std(OLS_residuals)`，其中 `weekly_gini = 2 × weekly_ROC_AUC − 1`；周等权、标准差 ddof=0，OLS 横轴为按 WEEK_NUM 排序的周序号。B/D 周号连续，因此使用原 WEEK_NUM 或平移后的序号得到相同斜率和残差。

来源：[官方评价说明](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview/tracks-and-awards)。
实现约定也对照了[参赛者公布的代码](https://note.erhoeht-x.jp/n/n6079318f58bc)。全部周保留，包括切分边界的不完整周；单类别周会令完整得分未定义，不删除或插值。

原核心报告使用周一开始的自然周，**与原始 WEEK_NUM 分组不同**。本次通过已缓存 train_base.parquet、index_in_full 恢复原周号，逐条核验日期、标签与唯一申请 ID。没有把元数据加入模型；恢复申请 ID 仍不等于验证客户独立性。

'''
    for split,a in audits.items():
        text+=f"- {split}：{a['n']:,} 条；{a['start']}—{a['end']}；WEEK_NUM {a['first_week']}—{a['last_week']}，{a['official_weeks']} 周；最少每周 {a['min_week_n']} 条、{a['min_week_events']} 个事件。\n"
    lgb=main[(main.model=='LightGBM')&(main.variant=='raw')].iloc[0]
    tabm=main[(main.model=='TabM')&(main.variant=='raw')].iloc[0]
    seed_raw=seed_summary[seed_summary.variant=='raw'].set_index('model')
    text+='\n## 结果解读\n\n'
    text+=f"D 的 seed=42：LightGBM 得分 {lgb.stability_score:.6f}，TabM 得分 {tabm.stability_score:.6f}，差值（TabM−LightGBM）为 {tabm.stability_score-lgb.stability_score:+.6f}。TabM 的主要优势来自较小的时间下降惩罚（{tabm.falling_penalty:.6f}，对比 {lgb.falling_penalty:.6f}），而非更高的平均周 Gini。\n\n"
    text+=f"全部固定 seed=42/0/1：LightGBM 均值±样本标准差为 {seed_raw.loc['LightGBM','stability_mean']:.6f}±{seed_raw.loc['LightGBM','stability_sd']:.6f}；TabM 为 {seed_raw.loc['TabM','stability_mean']:.6f}±{seed_raw.loc['TabM','stability_sd']:.6f}。这三个 seed 中 LightGBM 均值更高且波动更小，不能只选 seed=42 宣称 TabM 稳定胜出。没有对这个新增指标作显著性认证。\n\n"
    b=frames['B'].set_index('model')
    text+=f"同一个 B 上，原始 TabM 为 {b.loc['TabM / original','stability_score']:.6f}，冠军 LGB 特征适配 TabM 为 {b.loc['TabM / champion_lgb','stability_score']:.6f}，冠军三模型重建融合为 {b.loc['Champion reconstruction blend','stability_score']:.6f}。冠军 Cat 特征适配 TabM 虽曾取得较高整体 AUC，稳定性得分仅 {b.loc['TabM / champion_cat','stability_score']:.6f}，说明新指标会改变对开发方案的评价；本次没有据此重新选模型。\n"
    text+='\n## D：核心模型，seed 42\n\n'+md_table(main,cols)
    text+='\n\nraw/calibrated 的周 Gini 和最终得分相同是单调校准的预期结果，不能宣称校准提高排序或稳定性得分。\n\n![D 官方公式分解](official_metric_D.png)\n'
    text+='\n## D：全部匹配 seed\n\n'+md_table(seed_summary,['model','variant','n_seeds','stability_mean','stability_sd'])
    text+='\n\n这是 42/0/1 三个固定 seed 的均值和样本标准差，不是数据抽样置信区间。周等权会让样本较少的周具有同样权重；23 周单一时间窗口不能证明长期稳定性或统计显著优势。完整逐 seed 和逐周结果分别见 official_metric_D.csv、official_weeks_D.csv。\n'
    text+='\n## B：全部已完成开发方案\n\n'+md_table(frames['B'],cols)
    text+='''

这些是同一个 B 上的比较，B 只有 9 个官方周，且已被用于 early stopping 和多轮开发，因此不是独立验证。保留固定 score hack 行作为历史诊断，它不代表校准概率；没有调整其参数。ExcelFormer 未完成、S6E7 为 0/28，因此不把它们列为完成模型。

## 能否与 Kaggle 冠军比较

不能把这里的数值换算成榜单名次。我们评的是原比赛公开训练数据经 TabReD 取样后的本地时间留出；Kaggle 排名评的是隐藏测试集，样本、时间跨度、训练量和特征处理均不同。[冠军原文](https://www.kaggle.com/c/home-credit-credit-risk-model-stability/discussion/508337)所述未做 metric hacking 的最高 private 分数约 0.53 仅是背景，不能作为本地 D 的胜负门槛。相同公式不等于相同评测任务。

原版冠军仍缺私有拟合文件和最终列名单。此处“Champion reconstruction”是记录清楚的三模型重建，不是原版冠军精确复现。继承的全量无标签预处理、未知标签期限/成熟时间、未验证客户级独立性等限制仍成立。

## 复现与验证

在本目录用原有 core 虚拟环境执行：

```powershell
& ../../.venv/Scripts/python.exe -m pytest -q -o addopts= --basetemp=.pytest_tmp test_metric.py
& ../../.venv/Scripts/python.exe analyze.py
```

脚本从项目内既有缓存读取，拒绝原始 hash 不匹配；已完成后重跑只核验缓存。需要原本地私有缓存，公开仓库本身不包含行级输入。没有下载或启动模型。

每周 AUC 使用 sklearn 与独立的 Mann–Whitney rank-U 公式交叉计算；OLS 使用 polyfit 与中心化闭式解交叉计算。这是同一 agent 的不同计算路径自查，不是独立外部审计。verification.json 记录重算误差、旧 AUC/AP 一致性、raw/calibrated 一致性及保护文件哈希；RESULT_RECEIPT.json 保留代码、协议、输入和输出身份。
'''
    (HERE/'REPORT_ZH.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
