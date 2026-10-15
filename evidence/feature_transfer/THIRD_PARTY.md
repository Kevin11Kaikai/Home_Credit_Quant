# Sources and attribution

- **yuuniekiri**, [Home Credit LightGBM public kernel](https://www.kaggle.com/code/yuuniekiri/fork-of-home-credit-risk-lightgbm), version 16. Reuses unchanged `Pipeline`, `Aggregator`, and `agg_by_case` definitions, extracted via Python AST. The rest of the kernel is never executed.
- **yuuniekiri**, [Home Credit CatBoost inference public kernel](https://www.kaggle.com/code/yuuniekiri/fork-of-home-credit-catboost-inference), version 12. Reuses unchanged `Pipeline` and `Aggregator` definitions. The top-level CatBoost, model loading, submissions and data cleanup are never executed.
- [Champion write-up](https://www.kaggle.com/c/home-credit-credit-risk-model-stability/discussion/508337) provides context; its leaderboard score is not a target or an acceptance gate.
- [Home Credit — Credit Risk Model Stability 2024](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data). Only official training Parquet and field definitions downloaded, under the user's existing authorized competition access. No new terms accepted. Restricted raw records, row-level derived tables and predictions remain local.
- TabReD, official TabM, rtdl numerical embeddings and LightGBM versions/licenses remain those pinned in the parent project's `SOURCE_MANIFEST.json`, `THIRD_PARTY.md` and original frozen run.

The official kernel API did not return a code license. Public availability is
not treated as permission to redistribute. The local pinned copies retain
upstream comments and author attribution and are excluded from publication by
the local ignore file. Exact SHA256 and source container digests are recorded
in this experiment's `SOURCE_MANIFEST.json`; no claim is made that the local
Windows environment reproduces those containers byte for byte.

The official Kaggle CLI commands for every downloaded file are recorded in
`download_receipt.json`. No mirror, credential dump, upload or Kaggle submission
was used. No fraud-project data, weights or metrics enter these training inputs.
