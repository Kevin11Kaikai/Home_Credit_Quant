# 当前进度与复现入口（2026-10-15 发布）

核心项目已经完成；后续扩展仍按用户要求暂停。本次上传只整理代码与历史证据，没有恢复训练、重新推理、重新评价 D 或启动自动报告任务。

| 工作 | 实际完成情况 |
| --- | --- |
| Home Credit 2024 / TabReD 核心 MVP | CORE_COMPLETE；LightGBM、TabM 各 5 次拟合，最终匹配 seed=42/0/1 |
| 独立 C 校准与最终 D | 6 个校准器成功；一个冻结测试评价记录；贷款池、周度评价、bootstrap 和报告完成 |
| AbsBox | EXTENSION_COMPLETE；真实本地引擎执行；独立合成交易，已保存响应及勾稽结果 |
| 冠军特征迁移 | 6 次新拟合，另重放两个既有模型；仅复用开发集 B |
| 冠军三模型重建 | 15/15 拟合完成，固定权重融合完成；不等于原版冠军精确复现 |
| 原版冠军 | BLOCKED_DEPENDENCY：缺原作者私有拟合文件、最终列名单、完整训练配置；不是尚未接受赛事条款 |
| ExcelFormer | PAUSED_INCOMPLETE：3 个完整 epoch；第 4 个 epoch 的 offset=147456 检查点及 optimizer/RNG 仍在原本地工作区 |
| S6E7 | 0/28 个训练成员；代码、来源核查、合成测试及首折预处理已完成，不能声称已验证模型效果 |

## 已有结果应怎样阅读

核心测试 D 的 seed-42 AUC：LightGBM 0.862234、TabM 0.863624。原始模型差异的区间跨零，不能声称 Transformer 必胜；TabM 本身也不是 Transformer。独立 C 校准使两个模型在 D 上的整体 log loss/Brier 都变差，负结果完整保留。三组匹配 seed 的表格、概率校准、贷款池和真实 AbsBox 细节见[英文主报告](evidence/core/reports/PORTFOLIO_REPORT_FINAL.md)。

冠军三模型重建仅在重复使用的开发集 B 上得到 AUC=0.848150、AP=0.245916、log loss=0.143744。不能和 D 的数值直接比较。特征迁移、模型家族、拟合次数和选型规则不同，不能把整套配方的差异全部归因于某个特征或网络结构。

## 如何复现

1. **无需数据核验公开交付**：`python scripts/verify_public_release.py --summary`。只核验文件哈希和汇总表一致性，不伪装成模型重跑。
2. **从授权数据重新训练核心 MVP**：按 [REPRODUCE.md](REPRODUCE.md) 建独立环境，使用自己的 Kaggle 账号，下载唯一指定子集，在新 run_id 中执行 smoke/run/report/tests。
3. **后续配方**：源码与冻结配置公开，原始数据、行级派生文件、模型及无法确认再分发权限的第三方源码不上传。恢复上游源码的助手校验固定 SHA256；原版冠军私有文件仍不可得。完整混合依赖环境清单已保留，但本次没有从零重建并重跑所有神经模型。
4. **恢复原暂停任务**：必须明确另行要求，并在原本地工作区验证检查点及预算。上传仓库不是恢复指令，不会自动开训或启动 28 个 S6E7 成员。

公开包中的历史 manifest 路径经过本地路径脱敏；原始和公开文件 SHA256 的映射见 [PUBLIC_EXPORT_PROVENANCE.json](PUBLIC_EXPORT_PROVENANCE.json)。归档证据不能当成新工作目录的有效冻结状态。

所有损失都是 EAD=1、假设 LGD 下的 credit-event loss proxy；不是实际美元损失。AbsBox 是独立合成交易示例，不能把未知期限的事件概率转换成 annual CDR，更不能声称真实 CLO 投资业绩。
