# 从冠军特征流水线中借鉴什么

一条申请在历史申请、征信还款、税务、存款等原始表中可以对应多行。
公开方案按 `case_id` 将这些行聚合成申请级向量，再交给分类模型。
因此特征工程的主要价值是让模型直接看到历史水平、波动、极值和信息缺失。
这套输入可以用于不同的表格模型，作用并不限于树模型。

本次接收模型是现有官方 TabM，它属于表格神经网络，并非 Transformer。
我们实际检验的是这些特征向 TabM 的迁移；尚未运行 FT-Transformer。

| 可借鉴部分 | 公开代码实际做法 | 对神经网络输入的意义 | 需要避免的误读 |
|---|---|---|---|
| 历史水平与波动 | 对金额、逾期等字段做 max/min/mean/std；LGB 分支部分字段还有 sum/median | 提供不同尺度的历史摘要，降低仅靠网络学习汇总规则的负担 | 每种统计均有收益尚未单独消融 |
| 日期相对化 | 部分日期聚合后减去 `date_decision`，变成天数差 | 让输入表示相对申请的时间位置 | 未来日期可能是合同计划日期；不能仅凭符号认定泄漏或安全 |
| 类别历史 | Cat 分支对 M 类字段取 first/last/n_unique | 保留端点类别与类别变化数量 | first/last 是源代码指定顺序，不自动等于最早/最近时间 |
| 表特定聚合 | LGB 对征信、税务、person 等表使用不同表达式 | 比所有表统一 max 更有结构信息 | 这是完整表征包变化，不是单个特征的因果效果 |
| 缺失信息 | 本实验将 A 中变化的缺失模式显式附加为指示列 | 中位数填补后仍能区分真实中位数与缺失 | 这是本实验的迁移假设，不冒充冠军原样附加的特征 |
| 类别保留 | A-only 词表、missing=1、unknown=0，两个模型使用相同类别信息 | 使新类别处理可复现 | 与 TabReD 上游全量稀有类别合并不同，因此包间差异不全来自聚合 |

有真实定义的例子：`pmts_dpd_1073P` 是活跃合同的还款逾期天数；
`pmts_overdue_1140A` 是活跃合同的逾期还款；`amount_4527230A` 是税务登记的
扣税金额，**不是贷款 EAD**。`recorddate_4527225D` 是该税务记录日期。
这些含义来自本地官方 `feature_definitions.csv`；匿名列不补造金融含义。

一个不含真实客户数据的例子：同一申请三条数值记录为 `[1, 5, 9]`，
其 min=1、max=9、mean=5、样本 std=4。网络看到这些汇总量，可以区分
“一直处于中等水平”和“平均相同但波动更大”。这解释了候选信息的用途，
但不能代替实际实验中对效果的检验。

本次固定对照包括原特征、原特征加缺失标记、冠军 LGB 特征、冠军 Cat/DNN
特征。每种新输入同时训练 LightGBM 和 TabM。所有配置固定、seed=42，
不根据开发集结果再追加特征或挑选更有利的 seed。结果保留全部变体。
最后的结果报告会区分：是否帮助 TabM、是否也帮助 LightGBM、相同输入下
谁更好，以及证据是否足以延伸到独立测试期。

来源：[作者 LGB 代码](https://www.kaggle.com/code/yuuniekiri/fork-of-home-credit-risk-lightgbm)、
[作者 Cat 代码](https://www.kaggle.com/code/yuuniekiri/fork-of-home-credit-catboost-inference)、
[比赛数据说明](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data)。
具体版本、hash 和适配见本目录 `SOURCE_MANIFEST.json`、`ADAPTATIONS.md`。
