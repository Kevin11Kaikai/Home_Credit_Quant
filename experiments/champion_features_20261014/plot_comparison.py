"""Plot verified cached B results. No fitting, predictions or label access."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from common import HERE, read, write, sha, now, protected_hashes

OUT = HERE / 'visual_comparison'
ORDER = ['original', 'tabred_missing', 'champion_lgb', 'champion_cat']
LABELS = ['原始方法', '＋缺失标记', '冠军 LGB 特征', '冠军 Cat 特征']
MODELS = ['LightGBM', 'TabM']
METRICS = [('roc_auc', 'ROC-AUC ↑', 1), ('average_precision', 'Average precision (AP) ↑', 1),
           ('log_loss', 'Log loss ↓', -1), ('brier', 'Brier score ↓', -1)]
COLORS = {'LightGBM':'#256898', 'TabM':'#bd6a20'}
MARKERS = {'LightGBM':'o', 'TabM':'^'}


def verify_inputs():
    acceptance = read(HERE/'acceptance.json')
    assert acceptance['status'] == 'MATCHED_TRANSFER_COMPLETE'
    assert acceptance['full_champion_model'].startswith('NOT_REPRODUCED')
    for name, expected in acceptance['files'].items():
        assert sha(HERE/name) == expected, name
    assert read(HERE/'verification.json')['status'] == 'PASS'
    frame = pd.read_csv(HERE/'development_metrics_B.csv')
    assert len(frame)==8 and not frame.duplicated(['variant','model']).any()
    assert set(frame.seed)=={42} and set(frame.n)=={40596}
    assert set(zip(frame.variant,frame.model))=={(v,m) for v in ORDER for m in MODELS}
    assert np.isfinite(frame[[m[0] for m in METRICS]].to_numpy()).all()
    return frame.set_index(['variant','model'])


def save(fig, name):
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/(name+'.'+ext),dpi=180,facecolor='white')
    plt.close(fig)


def absolute_figure(frame):
    fig, axes = plt.subplots(2,2,figsize=(12.2,8.8))
    fig.subplots_adjust(left=.14,right=.975,top=.82,bottom=.14,wspace=.45,hspace=.48)
    fig.text(.5,.965,'冠军特征迁移：与原始方法的同条件比较',ha='center',fontsize=18,weight='bold')
    fig.text(.5,.925,'B 开发集：2020-01-01—2020-02-29 · 40,596 条 · seed=42 · 全部为未校准概率',
             ha='center',fontsize=11.5,color='#4b5563')
    handles=[Line2D([],[],color=COLORS[m],marker=MARKERS[m],linestyle='None',markersize=8,label=m) for m in MODELS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.9),ncol=2,frameon=False,fontsize=12)
    for ax,(metric,title,direction) in zip(axes.flat,METRICS):
        values=frame[metric].to_numpy()
        span=values.max()-values.min()
        ax.set_xlim(values.min()-span*.24,values.max()+span*.24)
        ax.set_ylim(3.55,-.55)
        ax.axhspan(-.43,.43,color='#f1f4f7',zorder=0)
        for y,v in enumerate(ORDER):
            l,t=(float(frame.loc[(v,m),metric]) for m in MODELS)
            ax.plot([l,t],[y,y],color='#c3cbd3',linewidth=2,zorder=1)
            for m,value,offset,align in [('LightGBM',l,8,'bottom'),('TabM',t,-8,'top')]:
                ax.scatter(value,y,color=COLORS[m],marker=MARKERS[m],s=52,zorder=3)
                ax.annotate(f'{value:.6f}',(value,y),xytext=(0,offset),textcoords='offset points',
                            ha='center',va=align,fontsize=10.8,color=COLORS[m])
        ax.set_yticks(range(4),LABELS,fontsize=11)
        ax.set_title(title+'  （'+('越高越好' if direction==1 else '越低越好')+'）',fontsize=12.5,pad=12)
        ax.set_xlabel('分数（局部范围）',fontsize=10.5)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f' if metric=='average_precision' else '%.4f'))
        ax.tick_params(axis='x',labelsize=10)
        ax.tick_params(axis='y',length=0,pad=7)
        ax.grid(axis='x',alpha=.18,zorder=0)
        for side in ['top','right','left']:ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#b3bbc3')
    fig.text(.14,.075,'复现范围：冠军公开特征的适配与迁移已完成；完整冠军模型、最终列名单与集成未复现。',fontsize=11,color='#334155')
    fig.text(.14,.039,'证据边界：B 已用于选型和 early stopping；单 seed，无独立测试结论。TabM 为表格神经网络。',fontsize=10.6,color='#64748b')
    fig.text(.975,.012,'来源：development_metrics_B.csv（已核验 hash）',ha='right',fontsize=9,color='#64748b')
    save(fig,'four_metric_comparison')


def delta_figure(frame):
    fig,ax=plt.subplots(figsize=(11.5,6.4))
    fig.subplots_adjust(left=.29,right=.98,bottom=.22,top=.79)
    fig.text(.5,.95,'哪些指标变好，哪些变差？',ha='center',fontsize=18,weight='bold')
    fig.text(.5,.902,'相对各模型的原始方法：Δ = 新特征分数 − 原始分数',ha='center',fontsize=12,color='#475569')
    ax.set_xlim(0,4);ax.set_ylim(6,0)
    rows=[];labels=[]
    for i,v in enumerate(ORDER[1:]):
        for j,model in enumerate(MODELS):
            row=2*i+j
            labels.append(LABELS[i+1]+' / '+model)
            for col,(metric,title,direction) in enumerate(METRICS):
                delta=float(frame.loc[(v,model),metric]-frame.loc[('original',model),metric])
                better=delta*direction>0
                label='分数更好' if better else ('分数更差' if delta else '相同')
                fill='#e0efe8' if better else ('#f4e2df' if delta else '#f1f3f5')
                ax.add_patch(Rectangle((col,row),1,1,facecolor=fill,edgecolor='white',linewidth=2))
                ax.text(col+.5,row+.37,f'{delta:+.6f}',ha='center',va='center',fontsize=12,color='#1e293b')
                ax.text(col+.5,row+.72,label,ha='center',va='center',fontsize=9.7,color='#52616b')
                rows.append(dict(variant=v,model=model,metric=metric,delta=delta,favourable_point_direction=better))
    ax.set_xticks(np.arange(4)+.5,[m[1] for m in METRICS],fontsize=11)
    ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=13)
    ax.set_yticks(np.arange(6)+.5,labels,fontsize=10.8)
    ax.tick_params(axis='y',length=0,pad=10)
    for s in ax.spines.values():s.set_visible(False)
    for y in [2,4]:ax.axhline(y,color='white',linewidth=6)
    fig.text(.06,.14,'绿色：点估计方向更好；红色：点估计方向更差。颜色不表示变化幅度，也不代表统计显著。',fontsize=11,color='#475569')
    fig.text(.06,.092,'同一 B 开发集 · seed=42 · 未重新训练或评价 D · “冠军特征适配”不等于完整冠军方案。',fontsize=10.5,color='#64748b')
    fig.text(.98,.035,'来源：development_metrics_B.csv；差值直接由缓存指标计算',ha='right',fontsize=9.5,color='#64748b')
    save(fig,'changes_vs_original')
    pd.DataFrame(rows).to_csv(OUT/'plotted_differences.csv',index=False)


def main():
    OUT.mkdir(exist_ok=True)
    font=Path('C:/Windows/Fonts/msyh.ttc')
    assert font.exists(),'Chinese font unavailable'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus':False,'svg.fonttype':'path','pdf.fonttype':42})
    inputs={n:sha(HERE/n) for n in ['development_metrics_B.csv','acceptance.json','verification.json','PROTOCOL_LOCK.json']}
    protected=protected_hashes()
    frame=verify_inputs()
    absolute_figure(frame)
    delta_figure(frame)
    assert inputs=={n:sha(HERE/n) for n in inputs}
    assert protected_hashes()==protected
    write(OUT/'plot_receipt.json',dict(status='PASS',generated_utc=now(),
          source_hashes=inputs,source_rows=8,seed=42,evaluation_split='B development',n=40596,
          metric_rounding=6,full_champion_model_reproduced=False,
          new_training=False,new_predictions=False,labels_loaded=False,original_artifacts_unchanged=True,
          code_sha256=sha(Path(__file__)),files={f.name:sha(f) for f in OUT.iterdir() if f.suffix in ['.png','.pdf','.svg','.csv']}))
    print('VISUAL_COMPARISON_READY',OUT)
    print('Verified 8 matched result rows; four metrics; no training or prediction generated.')


if __name__=='__main__':main()
