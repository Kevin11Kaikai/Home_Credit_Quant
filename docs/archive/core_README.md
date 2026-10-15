# Credit Risk under Temporal Shift

Calibrated Default Prediction and Portfolio Stress Analysis — independent public-data portfolio MVP.

**Delivery: CORE_COMPLETE / EXTENSION_COMPLETE.** The official Home Credit 2024 subset contains 381,664 rows and 696 base features. Each model completed three search candidates and two fixed-configuration replications; matched final seeds are 42, 0 and 1. Six independent-C calibrators fitted successfully before one frozen test evaluation. All 23 automated tests and 1,112 cached-output arithmetic comparisons passed. The core pipeline took 31.3 minutes; recorded model budgets were 2.40 minutes for LightGBM and 27.77 minutes for TabM, including the earlier smoke.

Seed-42 raw ROC-AUC was 0.862234 (LightGBM) and 0.863624 (TabM); raw model-difference bootstrap intervals span zero. Calibration worsened aggregate log loss and Brier for both models. At q=0.70 it worsened LightGBM's loss-proxy estimation error and slightly reduced TabM's. These negative and mixed results are retained. AbsBox actually ran against the pinned local Hastructure engine and passed cash reconciliation; its disclosed small loss-decomposition residuals remain in the report.

- [Run instructions](RUNME.md)
- [English portfolio report](runs/homecredit_temporal_v1_20261014/reports/PORTFOLIO_REPORT_FINAL.md)
- [Chinese summary](runs/homecredit_temporal_v1_20261014/reports/SUMMARY_FINAL_ZH.md)
- [Resume bullets, with actual status boundaries](runs/homecredit_temporal_v1_20261014/reports/RESUME_BULLETS_FINAL.md)
- [Reproducible technical report](runs/homecredit_temporal_v1_20261014/reports/PORTFOLIO_REPORT.md)
- [Primary/all-seed metrics](runs/homecredit_temporal_v1_20261014/metrics.csv), [weekly metrics](runs/homecredit_temporal_v1_20261014/weekly_metrics.csv)
- [Portfolio results](runs/homecredit_temporal_v1_20261014/portfolio_metrics.csv), [stress assumptions/results](runs/homecredit_temporal_v1_20261014/stress_scenarios.csv)
- [Training resource summary](runs/homecredit_temporal_v1_20261014/reports/training_summary.csv)
- [Machine-readable acceptance](runs/homecredit_temporal_v1_20261014/acceptance.json)
- [Data card](DATA_CARD.md), [official sources](SOURCE_MANIFEST.json), [attribution](THIRD_PARTY.md)
- [Real AbsBox tranche results](runs/homecredit_temporal_v1_20261014/absbox/tranche_summary.csv)

The fixed route is TabReD homecredit-default → LightGBM/official TabM → independent C calibration → normalized event-loss proxy pools. AbsBox is a separate synthetic senior/junior waterfall example. It is not another predictive model, a Home Credit cashflow conversion, a CLO valuation or an investment return backtest.

No private Anoris/JPMorgan data, weights, business metrics or internal systems are used. This workspace was initially empty and was not a Git repository. Nothing has been uploaded, published or pushed.

## Design boundaries

The package has an A/B-only training data object, A-only preprocessing, immutable protocol/configuration checks, positive-slope calibration, fixed within-week selection masks and one final test receipt per protocol. Predictions remain local, contain no labels, and join D labels only inside final evaluation. Tests use isolated synthetic fixtures; their numbers are not portfolio performance claims.

The inherited upstream full-data missingness/category processing cannot be undone here. The benchmark is **TabReD-derived with inherited preprocessing limitations**, not strictly point-in-time. Unknown target horizon and maturity prevent one-year-PD and tradeable-backtest claims. `index_in_full` is not a verified customer identifier.

The dependency environment is isolated in `.venv`. The package is intentionally small: `data`, `models`, `calibration`, `evaluation`, `portfolio`, `absbox_demo`, `reporting`, `pipeline`, and shared persistence utilities. Local JSON/CSV tracking records actual status. There is no website, cloud resource, external ensemble or additional dataset.
