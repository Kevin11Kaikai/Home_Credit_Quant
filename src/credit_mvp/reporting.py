from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .common import ROOT, Blocked, read_json, sha256, verify_freeze, write_json
from .data import LIMITATIONS


def markdown_table(frame):
    if frame.empty:
        return 'NOT EXECUTED: no empirical table is available.'
    columns=list(frame.columns)
    def cell(value):
        if pd.isna(value):
            return 'NA'
        return f'{value:.6f}' if isinstance(value,(float,np.floating)) else str(value)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def figures(run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=run_dir/'figures'
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi':140,'savefig.bbox':'tight','font.size':10})
    if (run_dir/'metrics.csv').exists():
        metrics=pd.read_csv(run_dir/'metrics.csv')
        main=metrics[metrics.seed==42].copy()
        main['label']=main.model+' '+main.variant
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,column in zip(axes,['log_loss','brier']):
            ax.bar(main.label,main[column],color='#3b7896')
            ax.set_title(column.replace('_',' ').title()+' (lower is better)')
            ax.tick_params(axis='x',rotation=40)
        fig.suptitle('Frozen temporal test — seed 42')
        fig.tight_layout()
        fig.savefig(directory/'probability_scores.png')
        plt.close(fig)
        reliability=pd.read_csv(run_dir/'reliability_bins.csv')
        reliability=reliability[reliability.seed==42]
        fig,ax=plt.subplots(figsize=(7,6))
        for (model,variant),part in reliability.groupby(['model','variant']):
            part=part[part.n>0]
            ax.plot(part.mean_probability,part.observed_event_rate,'o-',label=f'{model} {variant}')
            for _,r in part.iterrows():
                ax.annotate(str(int(r.n)),(r.mean_probability,r.observed_event_rate),fontsize=6)
        ax.plot([0,1],[0,1],'--',color='gray')
        ax.set(xlabel='Mean target-event probability',ylabel='Observed event rate',title='Reliability; labels show bin counts; C-derived edges')
        ax.legend(fontsize=8)
        fig.savefig(directory/'reliability.png'); plt.close(fig)
        weekly=pd.read_csv(run_dir/'weekly_metrics.csv')
        weekly=weekly[weekly.seed==42]
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        for (model,variant),part in weekly.groupby(['model','variant']):
            for ax,column in zip(axes,['roc_auc','log_loss']):
                ax.plot(pd.to_datetime(part.week),part[column],label=f'{model} {variant}')
                ax.set_ylabel(column)
        axes[0].legend(fontsize=8,ncol=2)
        axes[0].set_title('Weekly performance; single-class AUC = NA; all weeks retained')
        fig.autofmt_xdate(); fig.savefig(directory/'weekly_performance.png'); plt.close(fig)
        portfolio=pd.read_csv(run_dir/'portfolio_metrics.csv')
        fig,ax=plt.subplots(figsize=(7,5))
        for model,part in portfolio[(portfolio.seed==42)&(portfolio.variant=='raw')].groupby('model'):
            ax.plot(part.actual_retention,part.observed_event_rate,'o-',label=model)
        controls=pd.read_csv(run_dir/'random_control_summary.csv')
        ax.errorbar(controls.q,controls.observed_event_rate_mean,yerr=controls.observed_event_rate_std,label='Random (20 seeds, SD)',fmt='s--')
        ax.set(xlabel='Retention (random controls: nominal q)',ylabel='Observed target-event rate',title='Within-week equal-exposure selection; seed 42')
        ax.legend(); fig.savefig(directory/'retention_risk.png'); plt.close(fig)
        stress=pd.read_csv(run_dir/'stress_scenarios.csv')
        main=stress[(stress.seed==42)&(stress.q==.7)&(stress.variant=='calibrated')]
        fig,ax=plt.subplots(figsize=(7,5))
        for model,part in main.groupby('model'):
            ax.plot(part.scenario,part.normalized_stress_EL,'o-',label=model)
        ax.set(ylabel='Illustrative expected loss per unit exposure',title='Fixed q=0.70 pools; assumed odds and LGD scenarios')
        ax.legend(); fig.savefig(directory/'stress_loss.png'); plt.close(fig)
    if (run_dir/'absbox'/'tranche_summary.csv').exists():
        table=pd.read_csv(run_dir/'absbox'/'tranche_summary.csv')
        fig,ax=plt.subplots(figsize=(8,5))
        for tranche,part in table.groupby('tranche'):
            ax.plot(part.scenario,part.principal_repaid,'o-',label=tranche)
        ax.set(ylabel='Synthetic principal repaid',title='Real AbsBox engine — independent synthetic transaction')
        ax.legend(); fig.savefig(directory/'absbox_principal.png'); plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
        for scenario,color in [('ZeroPressure','#777777'),('Base','#237a8d'),('Adverse','#d7932c'),('Severe','#b94b48')]:
            for ax,tranche in zip(axes,['A1','B']):
                flow=pd.read_csv(run_dir/'absbox'/f'{scenario}_bond_{tranche}.csv',index_col=0)
                ax.step(pd.to_datetime(flow.index),flow.principal.cumsum(),where='post',label=scenario,color=color)
                ax.set_ylabel(f'{tranche} cumulative principal')
                ax.grid(alpha=.2)
        axes[0].set_title('Real local AbsBox cashflows — fixed synthetic senior/junior deal')
        axes[0].legend(ncol=4,fontsize=9)
        axes[1].set_xlabel('Payment date (synthetic contractual calendar)')
        fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(directory/'absbox_cashflows.png'); plt.close(fig)


def report(run_dir):
    """Read cached results only. Never train, calibrate, infer, select, or read y.npy."""
    run_dir=Path(run_dir).resolve()
    acceptance=read_json(run_dir/'acceptance.json')
    cfg=read_json(run_dir/'config.json')
    if (run_dir/'FROZEN_MANIFEST.json').exists():
        verify_freeze(run_dir/'FROZEN_MANIFEST.json')
    if (run_dir/'TEST_EVALUATION.json').exists():
        for name,expected in read_json(run_dir/'TEST_EVALUATION.json')['outputs'].items():
            if sha256(run_dir/name)!=expected:
                raise Blocked(f'Cached test output hash mismatch: {name}')
    audit=read_json(run_dir/('data_audit.json' if (run_dir/'data_audit.json').exists() else 'data_acquisition_status.json'))
    empirical=(run_dir/'metrics.csv').exists()
    source=read_json(ROOT/'SOURCE_MANIFEST.json')
    hw=read_json(run_dir/'hardware.json')
    main=pd.read_csv(run_dir/'metrics.csv') if empirical else pd.DataFrame()
    if empirical:
        main=main[main.seed==42][['model','variant','calibration_status','roc_auc','average_precision','log_loss','brier','mean_probability','observed_event_rate','probability_bias']]
    table=markdown_table(main)
    splits=markdown_table(pd.DataFrame([{'partition':k,**v} for k,v in audit.get('splits',{}).items()]).drop(columns=['index_sha256'],errors='ignore'))
    cal_text=[]
    if empirical:
        for model in ['LightGBM','TabM']:
            pair=main[main.model==model].set_index('variant')
            if 'raw' in pair.index and 'calibrated' in pair.index:
                if pair.loc['calibrated','calibration_status']!='FITTED':
                    cal_text.append(f'{model}: calibration failed; the labeled fallback is identical to raw probability and is not a calibrated result.')
                    continue
                delta=pair.loc['calibrated','log_loss']-pair.loc['raw','log_loss']
                db=pair.loc['calibrated','brier']-pair.loc['raw','brier']
                cal_text.append(f'{model}: calibrated minus raw log loss = {delta:+.6f}; Brier = {db:+.6f}. These are observed probability-score changes; inspect reliability and bias before claiming better calibration.')
    cal_summary=' '.join(cal_text) or 'Calibration on real C was NOT EXECUTED. There is no evidence of improvement or deterioration, and an engineering test is not a calibration result on Home Credit.'
    portfolio_text='Real loan-pool selection and loss evaluation were NOT EXECUTED because the dataset is absent. No retained event rate, expected loss or empirical loss-estimation error is claimed.'
    if empirical:
        p=pd.read_csv(run_dir/'portfolio_metrics.csv')
        portfolio_text=markdown_table(p[(p.seed==42)&(p.q==.7)][['model','variant','n','exposure','actual_retention','observed_event_rate','normalized_EL','normalized_observed_proxy','error_per_exposure']])
    ext=read_json(run_dir/'absbox'/'status.json') if (run_dir/'absbox'/'status.json').exists() else {'status':'NOT_ATTEMPTED_AFTER_INITIAL_REPORT','actual_engine_run':False}
    extension_text=f"Status: **{ext['status']}**. Real engine response obtained: **{ext.get('actual_engine_run',False)}**."
    if ext.get('reason'):
        extension_text+=' '+ext['reason']
    if (run_dir/'absbox'/'tranche_summary.csv').exists():
        extension_text+='\n\n'+markdown_table(pd.read_csv(run_dir/'absbox'/'tranche_summary.csv')[['scenario','tranche','principal_repaid','ending_balance','max_interest_due','ending_interest_due']])
    if (run_dir/'absbox'/'additional_reconciliation.json').exists():
        extension_text+='\n\nIndependent Decimal recomputation confirmed cash residuals of 0.00 in all four cases. The engine reports Default minus Recovery minus Loss residuals of 0.15, 0.15 and 0.12 under Base, Adverse and Severe; their cause is unverified and the raw values are preserved. The junior B interest field includes equity residual distributions and released overcollateralization, rather than pure contractual interest.'
    matched=read_json(run_dir/'matched_seeds.json') if empirical else {'n':0,'seeds':[]}
    text=f'''# Credit Risk under Temporal Shift: Calibrated Default Prediction and Portfolio Stress Analysis

## Question and actual status

This independent public-data portfolio project targets credit-event prediction, chronological validation, probability calibration, normalized loan-pool analysis and Python research engineering. It uses no private Anoris Capital or JPMorgan data, systems, weights or performance evidence. The empirical questions are deliberately limited: under a fixed temporal split and bounded search, how do LightGBM and official TabM compare in discrimination, probability quality and stability; and does a separately held-out positive-slope sigmoid calibrator improve later probability scores and pool loss estimates?

The core status is **{acceptance['core_status']}**. The extension status is **{acceptance['extension_status']}**. Formal empirical results available: **{empirical}**. Matched evaluated seeds: **{matched['n']}**, {matched['seeds']}. Missing items: {'; '.join(acceptance['missing']) or 'none'}. Results below describe executed work only. Synthetic contract checks are software evidence, not measured credit-risk performance. A blocked download does not establish a model failure or a negative empirical answer.

## Sources and data limitations

The intended dataset is TabReD homecredit-default, derived from Home Credit — Credit Risk Model Stability 2024, not the 2018 competition. TabReD is pinned to `{source['sources']['tabred']['commit']}` and TabM to `{source['sources']['tabm']['commit']}`. SOURCE_MANIFEST.json records retrieved official files, hashes, licenses and the exact single-subset acquisition command. The expected archive hash is independently recorded from the upstream registry. Kaggle authentication and competition access remain the user's responsibility; this project does not accept new terms or use alternative mirrors. Raw records and local predictions are excluded from public publication.

The inspected upstream script computes missingness and category cardinalities on the full feature collection before temporal splitting. It removes high-missingness and constant columns, filters some high-cardinality strings, and fits OrdinalEncoder with min_frequency=0.01 globally. It also uses full-table extrema for dtype downcasting. These are inherited/transductive preprocessing limitations. They are not, by themselves, demonstrated target-label leakage. Nevertheless, strict point-in-time preprocessing cannot be claimed. Refitting a vocabulary on A cannot undo rare categories already merged upstream. The source and current archive use different documented filenames, so exact artifact-generation lineage also remains unverified until inspected conservatively.

The source describes a default target, but an exact observation horizon and label maturity schedule have not been verified. Outputs are therefore dataset target-event probabilities or PD proxies, not one-year PDs. The documented metadata are date_decision, encoded as Unix epoch days, and index_in_full, a source row position. The latter is not a customer identifier; customer-level independence cannot be claimed. Exposures, realized recovery, default timing and contractual cashflows are not established by anonymous amount columns. These limitations prevent interpretation as a tradeable historical backtest.

## Temporal protocol and implementation

{splits}

The protocol keeps official train as A and official final test as D. The distinct validation dates are sorted and split at their midpoint: earlier complete dates form B and later complete dates form C. Boundaries depend only on metadata, never event rates. Strict A-before-B-before-C-before-D ordering and pairwise disjoint row indices are mandatory. Conflicting dates block evaluation instead of triggering a random replacement. This is a documented adaptation of TabReD, not an exact reproduction of its paper scores.

All newly fitted medians, constant masks, scaling, categorical vocabularies and numerical bins use A only. Both models receive the same underlying feature information. Metadata and labels cannot enter feature matrices. Unknown categories have an explicit code; missing categories use a separate code. LightGBM uses CPU native categorical features. TabM uses the official k=32 architecture, three width-256 blocks and 32-bin, dimension-16 piecewise-linear numerical embeddings. Each head contributes its own binary cross-entropy; inference averages sigmoid probabilities. No weighting, resampling or external ensemble is used.

Three frozen candidates per family use seed 42 and B AUC selection, with log loss and candidate ID resolving exact ties. The first candidate of each model precedes further searches. Additional fixed-configuration seeds follow 0 then 1, without seed selection. Limits are 20 minutes per LightGBM fit and two hours cumulatively, and 45 minutes per TabM fit and four hours cumulatively; interrupted results retain their actual status. The verified machine is {hw.get('gpu')}, with {hw.get('ram_total_bytes',0)/2**30:.2f} GiB RAM. CPU threads are limited to {cfg['cpu_threads']}. JSON/CSV tracking is local because this empty workspace had no MLflow configuration.

## Controlled comparison

{table}

The constant reference is mean(y_A). Average precision is named explicitly and is not relabeled as generic PR-AUC. Weekly tables retain small and poor-performing weeks, record sample counts and mark single-class AUC as NA. When enough consecutive real calendar weeks exist, 200 paired moving-block resamples use two-week blocks shared by both fixed models. Their intervals address short-range temporal sampling variation, not retraining uncertainty, unidentified customers or arbitrary longer dependencies. Seed means and standard deviations are reported only over matched seeds and do not replace data uncertainty.

## Independent calibration

{cal_summary}

The only calibrator is sigmoid(exp(theta) * logit(clipped raw probability) + b), optimized by unweighted binary log loss on C. Its identity initialization, bounds and iteration cap are frozen. Single-class or failed fits remain explicitly failed and return raw probabilities. C fitting scores are not presented as generalization results. C predictions also define deduplicated reliability-bin edges covering [0,1]; D labels never tune bins. Since positive slope preserves ranking mathematically, calibration cannot create new discrimination or improve a fixed-count pool's actual event rate. Floating-point clipping and saturation can introduce ties, so pool masks are always generated from raw probability and stable row ID and reused exactly.

## Loan-pool interpretation

{portfolio_text}

Retention fractions are fixed at 0.50, 0.70, 0.90 and 1.00, with 0.70 designated in advance. Selection occurs within each test week, retaining floor(q times the weekly count). This gives equal exposure across models; rounding is logged, including empty selected batches. Twenty fixed random seeds provide same-week, same-count controls. This is batch selection, not a sequential approval policy. Calibration changes estimated loss, not the records chosen. Smaller total losses from smaller pools are not interpreted as improvement; comparisons use loss per unit exposure.

Each record contributes EAD=1 normalized unit. Expected proxy loss sums probability times assumed LGD; the observed proxy sums event labels times the same assumed LGD. Neither quantity is an actual dollar loss or realized net credit loss. Base, Adverse and Severe use fixed odds multipliers 1.0/1.5/2.0 and LGDs 0.40/0.60/0.80 on unchanged pools. These illustrative parameters are not estimated macroeconomic or regulatory stresses. No stress-period accuracy, investment return, Sharpe ratio, VaR, expected shortfall or default-correlation model is claimed. Formula-driven loss increases are not independent research discoveries.

## AbsBox extension

{extension_text}

The extension uses only the official independent synthetic senior/junior example. Home Credit probabilities are never converted into annual CDR. Fixed Base/Adverse/Severe annual CDRs are 2%/5%/10%, annual CPRs 5%/10%/15%, recovery rates 60%/40%/20%, and monthly recovery lags 3/6/9. The API accepts annual rates directly. Only localhost is allowed. Requests, raw responses, engine/client versions and cashflows are retained when execution succeeds; a zero-pressure case checks initial collateral, liabilities and cash conservation. Unpaid interest balances are not repeatedly summed as new shortfalls. This demonstrates waterfall mechanics, not a complete CLO valuation or investment strategy.

## Reproducibility and limits

The package provides smoke, run and report commands plus automated contracts. A pretraining lock fixes decisions; a later frozen manifest hashes code, splits, preprocessing, checkpoints, calibrators and evaluation settings. A test-access receipt precedes reading D labels. One completion record belongs to one protocol; reporting reads cached artifacts and verifies their hashes. An observed D can never become unseen again after a bug fix. All factual numerical claims require the corresponding local artifacts. Authentication blockage is preserved as CORE_PARTIAL, while the independent extension can succeed or remain blocked without inventing missing predictive results.
'''
    reports=run_dir/'reports'
    reports.mkdir(exist_ok=True)
    (reports/'PORTFOLIO_REPORT.md').write_text(text,encoding='utf-8')
    summary=f"# 中文交付摘要\n\n核心：{acceptance['core_status']}；扩展：{acceptance['extension_status']}。\n\n"
    summary+=('真实主结果：\n\n'+table if empirical else 'Kaggle 认证阻塞，未获得真实数据；未执行正式训练、校准、测试或贷款池评价，不能给出真实日期、规模或模型优劣。')
    summary+='\n\n'+cal_summary+'\n\n'+extension_text+'\n\n上游存在全量无标签预处理统计；标签期限、成熟时间、客户独立性均未核验。归一化损失是事件标签乘假设 LGD，不是实际金额损失。\n'
    (reports/'SUMMARY_ZH.md').write_text(summary,encoding='utf-8')
    bullets=[
        'Implemented a reproducible temporal credit-risk research pipeline with isolated calibration, frozen evaluation artifacts, and matched-exposure portfolio contracts; real-data experimentation is pending Kaggle access.',
        ('Executed an official AbsBox synthetic senior/junior transaction through a local Hastructure engine under fixed stress scenarios, preserving requests, cashflows, and reconciliation evidence.' if ext['status']=='EXTENSION_COMPLETE' else 'Implemented normalized credit-event loss and synthetic waterfall scenario analysis with explicit data-provenance, calibration, and execution-status boundaries; no unexecuted performance gains claimed.')]
    if empirical:
        l=main[(main.model=='LightGBM')&(main.variant=='raw')]
        t=main[(main.model=='TabM')&(main.variant=='raw')]
        if len(l) and len(t):
            bullets[0]=f"Compared LightGBM and official TabM on a frozen temporal credit-event benchmark (seed 42 test AUC: {l.iloc[0].roc_auc:.4f} and {t.iloc[0].roc_auc:.4f}), with independent held-out calibration and equal-exposure loan-pool analysis."
    (reports/'RESUME_BULLETS.md').write_text('\n'.join('- '+b for b in bullets)+'\n',encoding='utf-8')
    (ROOT/'DATA_CARD.md').write_text('# Data card\n\n'+summary+'\n\n## Actual audit\n\n```json\n'+__import__('json').dumps(audit,indent=2)+'\n```\n',encoding='utf-8')
    figures(run_dir)
    write_json(run_dir/'results_availability.json',{'empirical_credit_results':empirical,'actual_test_evaluation':(run_dir/'TEST_EVALUATION.json').exists(),'absbox_actual_engine_run':ext.get('actual_engine_run',False),'report_words':len(text.split()),'missing':acceptance['missing']})
    return str(reports/'PORTFOLIO_REPORT.md')
