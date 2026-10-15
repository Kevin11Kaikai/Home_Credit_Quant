# Credit Risk under Temporal Shift

**Calibrated Default Prediction and Portfolio Stress Analysis**

An independent public-data credit-risk research project using Home Credit 2024 / TabReD, LightGBM, official TabM, independent-holdout calibration, normalized loan-pool analysis, and a separate synthetic AbsBox waterfall. This repository contains the current code, aggregate evidence, source/version records, and reproduction instructions. It does not imply work on Anoris/JPMorgan private data or internal systems, or actual CLO investment performance.

**Published 2026-10-15. Training remains paused.** Uploading this repository did not resume ExcelFormer or start S6E7.

| Workstream | Actual state |
| --- | --- |
| Original temporal MVP | **CORE_COMPLETE**: 5 fits per family, final seeds 42/0/1, six C-fitted calibrators, one frozen D evaluation |
| Independent synthetic AbsBox extension | **EXTENSION_COMPLETE**: actual local engine responses and reconciliation retained |
| Matched champion-feature transfer | **MATCHED_TRANSFER_COMPLETE**: six new seed-42 fits plus two existing-checkpoint B replays |
| Champion three-family reconstruction | **COMPLETE: 15 fits** (5 LightGBM, 5 DenseLight, 5 CatBoost) and fixed blending |
| Exact original champion | **BLOCKED_DEPENDENCY**: private fitted artifacts, exact final masks and training configuration unavailable |
| Official Gini stability analysis | **COMPLETE_POSTHOC_OFFICIAL_FORMULA**: cached D predictions for all three seeds and all 13 completed B rows; original `WEEK_NUM` verified |
| Fraud recipe / ExcelFormer transfer | **PAUSED_INCOMPLETE**: 3 completed epochs; fourth-epoch recovery position preserved locally |
| S6E7 FT-Transformer transfer | **NOT STARTED**: 0/28 trained members; code, source audit, synthetic tests and first-fold preprocessing exist locally |

Start with [中文进度说明](PROGRESS_ZH.md), [reproduction instructions](REPRODUCE.md), [machine-readable status](STATUS.json), and [publication boundaries](PUBLICATION_NOTES.md).

## How the core system works

Application features → A-only preprocessing → LightGBM or official TabM → B-only selection/early stopping → positive-slope sigmoid calibration fitted on C → frozen D probabilities → weekly metrics and fixed within-week low-risk pools. The constant reference is mean(y_A). TabM is an MLP-based ensemble, **not a Transformer**. Its training averages per-head BCE; inference averages per-head sigmoid probabilities.

The official subset has 381,664 application records and 696 base features. Metadata and row indices are excluded from model input. Four complete-date chronological partitions preserve the official final test:

| Split | Start | End | Applications |
| --- | --- | --- | --- |
| A | 2019-01-01 | 2019-12-31 | 267645 |
| B | 2020-01-01 | 2020-02-29 | 40596 |
| C | 2020-03-01 | 2020-04-30 | 17422 |
| D | 2020-05-01 | 2020-10-05 | 56001 |

## Core results: frozen D, seed 42

| Model | Probability | ROC-AUC ↑ | Average precision ↑ | Log loss ↓ | Brier ↓ |
| --- | --- | --- | --- | --- | --- |
| Constant | raw | 0.500000 | 0.021946 | 0.106961 | 0.021542 |
| LightGBM | raw | 0.862234 | 0.166495 | 0.085750 | 0.019990 |
| LightGBM | calibrated | 0.862234 | 0.166495 | 0.087655 | 0.020202 |
| TabM | raw | 0.863624 | 0.170869 | 0.086195 | 0.019971 |
| TabM | calibrated | 0.863624 | 0.170869 | 0.086383 | 0.020013 |

The slight raw AUC advantage of TabM is not evidence of a reliable win: its paired two-week moving-block bootstrap AUC-difference interval spans zero. Calibration **worsened aggregate log loss and Brier for both models**, despite being fitted on an independent C set. Positive-slope calibration preserves ranking; it does not improve AUC or change the fixed low-risk selection mask. Matched final seeds are 42, 0 and 1; seed variability is not data-sampling uncertainty.

![Frozen-D probability scores](evidence/core/figures/probability_scores.png)

- [English portfolio report](evidence/core/reports/PORTFOLIO_REPORT_FINAL.md), [中文总结](evidence/core/reports/SUMMARY_FINAL_ZH.md), [actual-result resume bullets](evidence/core/reports/RESUME_BULLETS_FINAL.md)
- [All-seed metrics](evidence/core/metrics.csv), [weekly metrics](evidence/core/weekly_metrics.csv), [reliability counts](evidence/core/reliability_bins.csv), [bootstrap intervals](evidence/core/bootstrap_summary.json)
- [Portfolio metrics](evidence/core/portfolio_metrics.csv), [stress scenarios](evidence/core/stress_scenarios.csv), [training resources](evidence/core/reports/training_summary.csv)

With EAD=1 and illustrative LGD, the pool module estimates **credit-event loss proxies**, not actual dollar losses. Selection is performed within each test week at q=0.50/0.70/0.90/1.00, with q=0.70 fixed as the main display. Equal-exposure random controls and raw/calibrated identical masks are retained. Stress losses rise by construction under the specified odds/LGD scenarios; these are not macro-calibrated stress forecasts.

The [AbsBox evidence](evidence/core/absbox/status.json) contains actual localhost engine 0.52.8 responses, synthetic requests, pool/bond cashflows, [reconciliation](evidence/core/absbox/reconciliation.json), [additional residual audit](evidence/core/absbox/additional_reconciliation.json), and [tranche summaries](evidence/core/absbox/tranche_summary.csv). No Home Credit rows or event probabilities were sent to the engine. This demonstrates cashflow mechanics, not CLO valuation or an investment backtest.

## Official competition formula: post-hoc analysis

The 2026-10-15 analysis applies Home Credit 2024's **Gini stability** formula to existing predictions: mean weekly Gini + 88 × min(0, weekly trend slope) − 0.5 × population standard deviation of the trend residuals. Weekly Gini is 2 × ROC-AUC − 1, and weeks receive equal weight. [Official evaluation](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview/tracks-and-awards).

The original `WEEK_NUM` was recovered and checked against cached dates and labels. It differs from the Monday-based calendar grouping in the original core report. Local D contains 56,001 applications in 23 official weeks (69–91); no week was dropped.

| Fixed seed | LightGBM | TabM |
| --- | ---: | ---: |
| 42 | 0.668858 | **0.692867** |
| 0 | **0.682922** | 0.662322 |
| 1 | **0.675957** | 0.642580 |
| Mean ± sample SD, all three seeds | **0.675912 ± 0.007032** | 0.665923 ± 0.025336 |

Raw and calibrated results are identical, as expected from ranking-preserving calibration. TabM's seed-42 advantage mainly comes from its smaller downward-trend penalty; LightGBM has the higher mean and lower between-seed variation across all three fixed seeds. Seed SD is not a sampling confidence interval, and no statistical superiority is claimed for this added metric.

On reused development B (nine official weeks), original TabM scores 0.683865, champion-LGB-feature TabM 0.689479, and the three-family reconstruction blend 0.686447. All 13 completed B rows, including negative results, are retained. These comparisons did not trigger model or seed reselection.

**These are local scores, not Kaggle leaderboard results.** B/D come from the public training population processed through TabReD, rather than the competition's hidden test set. The formula was added after inspecting the original results and was not the preregistered primary metric. The original frozen evaluation, inherited preprocessing limitations, and paused training state remain intact.

![Official-week Gini and local stability scores](analyses/official_metric_20261015/official_metric_D.png)

- [Detailed Chinese report](analyses/official_metric_20261015/REPORT_ZH.md), [all D scores](analyses/official_metric_20261015/official_metric_D.csv), [all B scores](analyses/official_metric_20261015/official_metric_B.csv), [seed summary](analyses/official_metric_20261015/seed_summary_D.csv)
- [Weekly D aggregates](analyses/official_metric_20261015/official_weeks_D.csv), [week-mapping audit](analyses/official_metric_20261015/week_mapping_audit.json), [protocol](analyses/official_metric_20261015/PROTOCOL.json), [calculation verification](analyses/official_metric_20261015/verification.json)
- [Implementation and tests](analyses/official_metric_20261015/metric.py), [reproduction guide](REPRODUCE.md#7-official-formula-analysis), [publication-update validation](PUBLICATION_UPDATE_VALIDATION_20261015.json)

## Follow-up training results: reused development B only

Follow-up training and recipe development used A/B only; the separate official-formula analysis above later rescored existing D predictions without fitting or reselection. These B scores must not be mixed with the D table above. Feature transfer used the same fixed selected model settings and seed 42. The complete three-family **reconstruction** achieved B ROC-AUC **0.848150**, AP **0.245916**, and log loss **0.143744**. Its published earlier-week score heuristic worsened B AUC to 0.814947 and is retained as a negative result, not a calibrated PD.

![Completed B recipe comparison](evidence/full_solutions/reports/comparison_B.png)

- [All feature-transfer controls, including missing-indicator variants](evidence/feature_transfer/development_metrics_B.csv), [feature changes](evidence/feature_transfer/FEATURE_TRANSFER_GUIDE_ZH.md)
- [All completed B recipe rows](evidence/full_solutions/reports/comparison_B.csv), [three-family component scores](evidence/full_solutions/results/champion/metrics.csv)
- [Exact-champion blocker](evidence/full_solutions/CHAMPION_REPRODUCTION_AUDIT.json), [official private-input observation](evidence/full_solutions/SOURCE_AUDIT_ADDENDUM.json)
- [Recipe adaptations](evidence/full_solutions/REPRODUCTION_BOUNDARIES.md), [S6E7 source note](evidence/full_solutions/S6E7_SOURCE_NOTE.md), [pause receipt](evidence/full_solutions/PAUSE_RECEIPT.json)

The requested fraud repository's primary recipe is ExcelFormer with ordered target encoding and MI-weighted feature mixing. Its interrupted learning curve is historical progress, not a completed comparison. S6E7's 7×4-member FT-Transformer/PLR-lite/exact-value-TE recipe is implemented but has **no completed credit-risk model or score**. The source competition's 0.95084 balanced accuracy is not a Home Credit result.

## Reproducibility and limits

```powershell
python scripts/verify_public_release.py --summary
```

This checks public bytes and aggregate consistency without data or a GPU, including recalculating the stability formula from the published weekly Gini aggregates. It cannot verify individual predictions or weekly AUC from withheld records. [REPRODUCE.md](REPRODUCE.md) separately explains pinned environments, authorized Kaggle acquisition, synthetic tests, fresh core runs, optional local AbsBox, optional source restoration, and the paused follow-up. Model checkpoints and row outputs are withheld; this export is not a bitwise checkpoint replay package. The initial publication verification is recorded in [PUBLICATION_VALIDATION.json](PUBLICATION_VALIDATION.json), with this update recorded [separately](PUBLICATION_UPDATE_VALIDATION_20261015.json).

The upstream preprocessing used global unlabeled feature statistics before splitting, including rare-category merging. Train-only processing here cannot undo that. Target horizon, maturity, customer-level independence and as-of feature availability remain unverified. The correct description is a **TabReD-derived temporal benchmark with inherited preprocessing limitations**; predictions are dataset event probabilities / PD proxies, not one-year PDs or tradeable backtest signals.

See [DATA_CARD.md](DATA_CARD.md), [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json), [THIRD_PARTY.md](THIRD_PARTY.md), and [export provenance](PUBLIC_EXPORT_PROVENANCE.json). Original data, secrets, row-level predictions, checkpoints, and unlicensed upstream source snapshots are not published. No training/monitoring automation is installed by this repository.
