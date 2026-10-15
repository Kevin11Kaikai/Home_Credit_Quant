"""Cache-only reports. Never fits, chooses a seed, or reads C/D labels."""
from __future__ import annotations
import argparse,math
import numpy as np,pandas as pd
from common import *
from data import labels,mapping

def collect():
    rows=[];predictions={};source_hashes={}
    old=pd.read_csv(PREVIOUS/'development_metrics_B.csv')
    # Keep all frozen controls in the table; the figure uses predeclared rows.
    for _,r in old.iterrows():
        name=f"{r['model']} / {r['variant']}"
        row=r.to_dict();row.update(name=name,recipe='Previously completed fixed control',evaluation='B reused development',status=r['status']);rows.append(row)
        p=PREVIOUS/('baseline_replay/'+r['model']+'.npy' if r['variant']=='original' else 'fits/'+r['variant']+'/'+r['model']+'/predictions_B.npy')
        if p.exists():predictions[name]=np.load(p);source_hashes[str(p)]=sha(p)
    champ=HERE/'results/champion'
    if (champ/'receipt.json').exists():
        receipt=saved(champ)
        for _,r in pd.read_csv(champ/'metrics.csv').iterrows():
            name=r['model'];row=r.to_dict();row.update(name=name,model=name,recipe='Declared three-family reconstruction; not exact champion',seed=42,status=receipt['status'],evaluation='B reused development');rows.append(row)
            key={'Champion reconstruction lgb':'lgb_B.npy','Champion reconstruction dense':'dense_B.npy','Champion reconstruction cat':'cat_B.npy','Champion reconstruction blend':'ensemble_B.npy','Champion reconstruction + score hack (not PD)':'score_hack_B.npy'}[name]
            predictions[name]=np.load(champ/key);source_hashes[str(champ/key)]=sha(champ/key)
    excel=HERE/'runs/excel'
    if (excel/'receipt.json').exists():
        r=saved(excel);name='Fraud recipe ExcelFormer transfer';row=dict(name=name,model='ExcelFormer',recipe='Ordered TE + MI feature mix; adapted credit features',seed=42,status=r['status'],evaluation='B checkpoint selection by AP',new_training_seconds=r['seconds'],**r['metrics']);rows.append(row);predictions[name]=np.load(excel/'p_valid.npy');source_hashes[str(excel/'p_valid.npy')]=sha(excel/'p_valid.npy')
    s6=HERE/'results/s6e7'
    if (s6/'receipt.json').exists():
        r=saved(s6);name='S6E7 FT-Transformer transfer 7x4';row=pd.read_csv(s6/'metrics.csv').iloc[0].to_dict();row.update(name=name,model='FT-Transformer',recipe='Complete 7 folds x 4 members x 16 epochs',seed='42,43,44,45 per fold',status=r['status'],evaluation='B, no checkpoint or configuration selection');rows.append(row);predictions[name]=np.load(s6/'ensemble_B.npy');source_hashes[str(s6/'ensemble_B.npy')]=sha(s6/'ensemble_B.npy')
    return pd.DataFrame(rows),predictions,source_hashes

def status():
    state=read(HERE/'campaign_state.json') if (HERE/'campaign_state.json').exists() else {'status':'NOT_STARTED'}
    import psutil
    running=state['status']=='RUNNING' and psutil.pid_exists(state.get('pid',-1))
    exact=read(HERE/'CHAMPION_REPRODUCTION_AUDIT.json')['status']
    champion=read(HERE/'results/champion/receipt.json')['status'] if (HERE/'results/champion/receipt.json').exists() else 'INCOMPLETE'
    ex=read(HERE/'runs/excel/receipt.json') if (HERE/'runs/excel/receipt.json').exists() else {}
    members=[]
    for f in range(7):
        for m in range(4):
            p=HERE/'runs'/f's6e7_fold{f}_member{m}/receipt.json'
            r=read(p) if p.exists() else {}
            members.append(dict(fold=f,member=m,seed=42+m,status=r.get('status','NOT_FINISHED'),epochs=r.get('completed_epochs',0)))
    count=sum(m['status']=='COMPLETE' and m['epochs']==16 for m in members)
    s6='COMPLETE_7_FOLDS_4_MEMBERS' if (HERE/'results/s6e7/receipt.json').exists() and count==28 else ('RUNNING' if running and state.get('family')=='s6e7' else 'INCOMPLETE')
    tests=read(HERE/'tests_receipt.json') if (HERE/'tests_receipt.json').exists() else {'status':'NOT_RECORDED'}
    full=champion=='THREE_FAMILY_RECONSTRUCTION_COMPLETE' and ex.get('status')=='COMPLETE' and s6=='COMPLETE_7_FOLDS_4_MEMBERS'
    lock=read(HERE/'PROTOCOL_LOCK.json');preserved=all(sha(p)==h for p,h in lock['protected_artifacts'].items())
    integrity=sha(HERE/'config.json')==lock['config_sha256'] and all(sha(HERE/p)==h for p,h in lock['code'].items())
    counts={kind:sum((HERE/'runs'/f'champion_{kind}_fold{f}/receipt.json').exists() for f in range(5)) for kind in ['lgb','cat','dense']}
    return dict(overall='REQUEST_PARTIAL_EXACT_CHAMPION_BLOCKED' if full else ('REQUEST_RUNNING' if running else 'REQUEST_PARTIAL'),exact_champion=exact,champion_reconstruction=champion,champion_completed_fits=counts,champion_required_fits=15,fraud_transfer=ex.get('status','RUNNING' if running and state.get('family')=='excel' else 'NOT_FINISHED'),s6e7_transfer=s6,s6e7_completed_members=count,s6e7_required_members=28,s6e7_members=members,process=state,process_alive=running,tests=tests,original_mvp_preserved=preserved,protocol_integrity=integrity,new_evaluation_scope='A OOF and reused B development only',new_C_D_accessed=False,updated_utc=now())

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
    out=HERE/'reports';out.mkdir(exist_ok=True)
    table,preds,hashes=collect();a=status();write(HERE/'acceptance.json',a)
    table.to_csv(out/'comparison_B.csv',index=False)
    y,yb=labels();mp=mapping().iloc[len(y):].reset_index(drop=True);weekly=[];rechecks=[]
    for name,p in preds.items():
        m=metrics(yb,p);r=table.loc[table.name==name].iloc[0]
        for key,value in m.items():
            if key in r and pd.notna(r[key]):assert abs(float(r[key])-value)<1e-7,(name,key)
        rechecks.append(dict(name=name,**m))
        for week,ix in mp.groupby('WEEK_NUM').groups.items():
            ix=np.array(list(ix));yy=yb[ix];pp=p[ix]
            weekly.append(dict(name=name,week=int(week),n=len(ix),single_class=len(np.unique(yy))<2,auc=roc_auc_score(yy,pp) if len(np.unique(yy))==2 else np.nan,log_loss=log_loss(yy,pp,labels=[0,1]),small_week=len(ix)<200))
    pd.DataFrame(weekly).to_csv(out/'weekly_B.csv',index=False);pd.DataFrame(rechecks).to_csv(out/'recomputed_B.csv',index=False)
    # Display fixed baselines and only completed, named full recipes. Never
    # present a subset of 28 members as the full S6E7 ensemble.
    selected=['LightGBM / original','TabM / original','LightGBM / champion_lgb','TabM / champion_lgb','LightGBM / champion_cat','TabM / champion_cat','Champion reconstruction blend','Fraud recipe ExcelFormer transfer','S6E7 FT-Transformer transfer 7x4']
    # A budget-stopped single model remains visible in the results table, with
    # its PARTIAL status, but is not drawn as a completed recipe comparison.
    selected=[s for s in selected if s in preds and (
        s!='Fraud recipe ExcelFormer transfer' or
        table.loc[table.name==s,'status'].iloc[0]=='COMPLETE')]
    labels_short={'LightGBM / original':'Original LightGBM','TabM / original':'Original TabM','LightGBM / champion_lgb':'LGB feature set + LightGBM','TabM / champion_lgb':'LGB feature set + TabM','LightGBM / champion_cat':'Cat feature set + LightGBM','TabM / champion_cat':'Cat feature set + TabM','Champion reconstruction blend':'Champion reconstruction blend','Fraud recipe ExcelFormer transfer':'Fraud recipe ExcelFormer','S6E7 FT-Transformer transfer 7x4':'S6E7 FT-Transformer (7 x 4)'}
    t=table.set_index('name').loc[selected]
    fig,axs=plt.subplots(2,2,figsize=(13,max(8,len(selected)*1.1)),layout='constrained')
    colors=['#a3aab5','#5f8eaa','#aab8d5','#63b8a3','#a5b4fc','#439a86','#e4a04d','#c66b90','#7866b0'][:len(selected)]
    for ax,key,title in zip(axs.ravel(),['roc_auc','average_precision','log_loss','brier'],['ROC-AUC (higher better)','Average precision (higher better)','Log loss (lower better)','Brier score (lower better)']):
        values=t[key].astype(float).to_numpy();ax.barh(np.arange(len(t)),values,color=colors);ax.set_yticks(np.arange(len(t)),[labels_short[n] for n in selected],fontsize=9);ax.invert_yaxis();ax.set_title(title);ax.set_xlim(0,max(values)*1.17)
        for j,v in enumerate(values):ax.text(v+max(values)*.015,j,f'{v:.5f}',va='center',fontsize=9)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Home Credit: reused development B, same 40,596 applications\nRecipe comparison; not a new independent test or Kaggle leaderboard reproduction',fontsize=13)
    fig.savefig(out/'comparison_B.png',dpi=170);fig.savefig(out/'comparison_B.svg');plt.close(fig)
    w=pd.DataFrame(weekly);fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for n in selected:
        v=w[w.name==n];ax.plot(v.week,v.auc,marker='o',ms=3,label=labels_short[n])
    ax.set(xlabel='Verified competition WEEK_NUM',ylabel='Weekly ROC-AUC',title='Temporal development B: no weekly retraining');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.savefig(out/'weekly_B.png',dpi=160);plt.close(fig)
    budget=read(HERE/'budget.json') if (HERE/'budget.json').exists() else {}
    report=['# 完整方案复现与迁移：实际运行状态','',f"更新时间：{a['updated_utc']}",'',f"当前总体状态：**{a['overall']}**。原版冠军：**{a['exact_champion']}**；三模型重建版：**{a['champion_reconstruction']}**；ExcelFormer 迁移：**{a['fraud_transfer']}**；S6E7：**{a['s6e7_transfer']}**，完成 **{a['s6e7_completed_members']}/28** 个 16-epoch 成员。",'',
      '输入是现有官方 Home Credit 2024 缓存产生的申请记录与历史聚合特征。A 为 2019-01-01 至 2019-12-31，共 267,645 条；B 为 2020-01-01 至 2020-02-29，共 40,596 条。新增运行没有读取 C/D。所有下表结果都是已复用的 B 开发集结果，不能称为新独立测试。','',
      '冠军重建版先在 A 内按真实周分成 5 折，各自拟合 LightGBM、CatBoost、官方 DenseLight，再按公开的 0.40/0.36/0.24 权重平均。原作者最终列名单、训练超参数和拟合文件不可得；因此这只是已明确参数的三模型重建。源帖的提前周减 0.03 是比赛分数启发式，其输出另存，不能当作校准后的 PD 或用于贷款损失。','',
      'Fraud 仓库的主方案是 ExcelFormer。迁移保留按历史标签块编码类别、训练集 MI 排序及特征混合 BCE；当前数据缺少可验证的客户 UID、银行卡与地址字段，所以没有伪造这些实体特征。初始 16 epoch 后从该阶段最佳模型继续训练，最多再 25 epoch。它按 B 的 AP 选 checkpoint；这与基线的 B-AUC 选型规则不同，不能把差异纯粹归因于网络架构。','',
      'S6E7 保留 7 折 × 4 成员、PLR-lite 和折内 exact-value target encoding。内部编码还使用 5 折 cross-fitting，预处理只拟合各外层训练折。三分类源任务迁移为二分类编码/BCE；每个成员固定训练 16 epoch，取最后一轮。原方案的类别先验修正另存为分类诊断，不用于概率评分。未满 28 个成员时不报告为完整 S6E7 结果。','',
      '下表保留此前全部固定对照以及已经完成的新结果。没有根据 B 结果追加候选或挑最好 seed。', '',table[[c for c in ['name','status','roc_auc','average_precision','log_loss','brier'] if c in table]].to_markdown(index=False,floatfmt='.6f'),'',
      '![开发集概率与区分指标](comparison_B.png)','', '![逐周开发集表现](weekly_B.png)','',
      '限制：该比较存在开发集重复使用；原上游 TabReD 预处理包含跨切分无标签统计，新增特征筛选也继承之前按 A 拟合的列过滤，A-OOF 不应称为完全独立验证。标签期限、成熟时间和客户级独立性仍不可核实。各方法的特征表达、训练次数、选型规则和算力不同；这是完整配方比较，不是单一架构因果实验。没有将预测概率称为一年期 PD。','',
      '来源和依赖阻塞见 `SOURCE_MANIFEST.json`、`CHAMPION_REPRODUCTION_AUDIT.json`、`SOURCE_AUDIT_ADDENDUM.json`、`THIRD_PARTY.md`。各配方的完整性与适配边界见 `REPRODUCTION_BOUNDARIES.md`；冻结设置见 `config.json` 和 `PROTOCOL_LOCK.json`。累计耗时见 `budget.json`；单模型指标、最优轮数、恢复检查点见 `runs/*/receipt.json`。当前图表从本地预测缓存重算，无训练副作用；预算中断的单模型仍保留在表格中，不能冒充完成的配方。','',
      '累计分族运行秒数：`'+json.dumps(budget.get('used',{}),ensure_ascii=False)+'`。该台机器仍在运行时，以 `campaign_state.json` 和各 trial 的 `progress.json` 为准。']
    (out/'FULL_SOLUTIONS_REPORT_ZH.md').write_text('\n'.join(report),encoding='utf-8')
    write(out/'report_receipt.json',dict(created_utc=now(),source_prediction_hashes=hashes,cache_only=True,all_reported_metrics_recomputed=True,independent_recalculation=False,rows=len(table),files=seal(out,['comparison_B.csv','recomputed_B.csv','weekly_B.csv','comparison_B.png','comparison_B.svg','weekly_B.png','FULL_SOLUTIONS_REPORT_ZH.md'])))
    print(json.dumps({k:v for k,v in a.items() if k not in ['s6e7_members','tests']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
