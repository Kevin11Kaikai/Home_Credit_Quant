# Data card: TabReD-derived Home Credit 2024

The official `homecredit-default` subset was acquired on 2026-10-14 after the user completed Kaggle OAuth and competition access was verified. It contains **381,664 rows and 696 base features**: 612 numerical, 82 categorical and 2 binary. No 2018 Home Credit data are used.

Each row is a benchmark loan-application/case record. Feature names are absent from this archive; anonymous `num_####`, `cat_####` and `bin_####` identifiers do not imply economic meaning. Amount-like features are not assumed to be EAD. The source retains day-of-week, day-of-month and day-of-year features, but their exact anonymous column positions are not recovered.

`x_meta` contains decision-date Unix epoch days and `index_in_full`, an upstream row position. Neither metadata column enters either model. `case_id` and a verified customer identifier are unavailable. Customer-level independence is therefore unverified. All extracted input-file hashes, dimensions and overlap checks are in `runs/homecredit_temporal_v1_20261014/data_audit.json`.

| Partition | First date | Last date | Rows |
| --- | --- | --- | ---: |
| A | 2019-01-01 | 2019-12-31 | 267,645 |
| B | 2020-01-01 | 2020-02-29 | 40,596 |
| C | 2020-03-01 | 2020-04-30 | 17,422 |
| D | 2020-05-01 | 2020-10-05 | 56,001 |

B/C divide the ordered distinct official validation dates at their midpoint, without splitting a day. Official D indices are preserved. All four sets are disjoint and strictly ordered. The artifact spans 2019-01-01 through 2020-10-05; the full raw-table historical range remains unverified. Label-free hashing found 0 duplicate feature rows, 0 repeated source-row positions, and no cross-partition feature-hash duplicates. Hash checks are not proof of customer independence.

The target is the dataset credit event, described as default upstream. Its exact horizon, observation window and label maturity cannot be verified. Predictions are **dataset target-event probabilities / PD proxies**, never one-year PDs. The experiment is not a tradeable historical backtest.

This is a **TabReD-derived temporal benchmark with inherited preprocessing limitations**. Pinned upstream preprocessing uses full-data missingness, constants and string cardinalities for filtering, full-table dtype extrema, and `OrdinalEncoder(min_frequency=0.01)` before temporal splitting. These are inherited/transductive preprocessing limitations, not demonstrated target-label leakage. Exact archive-generation lineage and raw-table as-of feature availability remain unverified. A-only re-encoding cannot reconstruct merged rare categories. All new preprocessing is fitted only on A; 19 A-constant numerical/binary columns are removed for both models.

Verified exposure amounts, recovery rates, default times and contractual cashflows are unavailable. Portfolio calculations use EAD=1 normalized unit and assumed LGD. The observed proxy means **event label times assumed LGD**, not realized net loss or dollars. AbsBox uses a wholly separate synthetic pool; no Home Credit records or probabilities are sent to its engine.

The official downloader checks the registry archive SHA256 `fdcdcfe9b67e9ac8fae0291ee0aa8b492e41c123f99c9d3cecae2b78602c7f30`. Its temporary archive is removed by upstream cleanup; individual extracted-file hashes are retained. Kaggle competition/data-use terms apply independently of software licenses. Raw data, labels, row-level predictions and credentials remain local and are excluded from publication. No new terms were automatically accepted.
