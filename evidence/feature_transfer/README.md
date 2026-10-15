# Home Credit 2024 champion feature transfer

This is a separately authorized, bounded follow-up to the completed frozen MVP.
It reproduces published feature expressions and tests their transfer to the
existing official TabM model, with matched LightGBM controls. TabM is not a
Transformer architecture. No additional neural architecture is introduced.

The two public kernels are inference releases. Their final column masks depend
on fitted model artifacts not included in the retrieved code. This experiment
does **not** claim to reproduce the champion's complete ensemble, final score,
or exact 644/661-column matrices. It does not apply leaderboard metric tricks.

## Data flow

Official Home Credit 2024 training Parquet tables → raw application IDs matched
to the existing TabReD A/B rows → author's per-application aggregation expressions
→ date offsets relative to the application → A-only filtering, median filling,
scaling, explicit missing indicators and unknown-category encoding → TabM or
LightGBM → development B predictions. Application IDs and dates used for alignment
never become direct model inputs. Author calendar month/weekday features remain.

A contains 267,645 applications from 2019-01-01 through 2019-12-31. B contains
40,596 applications from 2020-01-01 through 2020-02-29. Dates and A/B labels match
the existing benchmark exactly. No C/D labels or official competition test data
are used. B has already been used for model selection; all results are exploratory
development evidence, not a new unseen-test evaluation. Target horizon, maturity,
source record availability and customer-level identity remain unverified.

## Fixed comparison

| Input | LightGBM | TabM |
|---|---|---|
| Original TabReD representation | Replay existing seed-42 checkpoint on B | Same |
| Original + explicit missing indicators | One new fit | One new fit |
| Author LightGBM feature expressions | One new fit | One new fit |
| Author Cat/DNN feature expressions | One new fit | One new fit |

Every new fit uses seed 42, the original selected L3/T2 configuration, the same
A/B rows and AUC checkpoint selection. There is no new hyperparameter search,
seed selection, ensemble, calibration, D evaluation or portfolio re-estimation.
Each representation is supplied to both model families. The champion inputs are
replacement representations; effects combine aggregation, column and category
changes and must not be attributed to one feature without further evidence.

New limits: CPU threads 4; LightGBM 30 minutes total and 20 minutes per fit;
TabM 90 minutes total and 45 minutes per fit; worker sequence 2 hours wall time.
The budget ledger conservatively includes process, input and binning time.
Model training has its own deadline, and subprocesses allow up to 300 seconds
for serialization/cleanup. Existing MVP budgets and artifacts are never reset.
Budget-limited best checkpoints remain partial, not converged fits.

## Reproduction (PowerShell, from project root)

```powershell
Set-Location '${SOURCE_WORKSPACE}'
# Official source versions, hashes and original download commands are recorded
# in SOURCE_MANIFEST.json and download_receipt.json. Reuse verified local data.
& 'experiments\champion_features_20261014\.venv_fe\Scripts\python.exe' 'experiments\champion_features_20261014\features.py'
& '.venv\Scripts\python.exe' 'experiments\champion_features_20261014\training.py' prepare
& 'experiments\champion_features_20261014\.venv_fe\Scripts\python.exe' -m pytest -q 'experiments\champion_features_20261014\test_features.py'
& '.venv\Scripts\python.exe' -m pytest -q 'experiments\champion_features_20261014\test_training.py'
& '.venv\Scripts\python.exe' 'experiments\champion_features_20261014\training.py' run
& '.venv\Scripts\python.exe' 'experiments\champion_features_20261014\training.py' report
& '.venv\Scripts\python.exe' 'experiments\champion_features_20261014\verify_completed.py'
& '.venv\Scripts\python.exe' 'experiments\champion_features_20261014\write_report.py'
```

`run` reuses finished fits; it never silently retries an interrupted formal fit.
`report` reads saved B predictions. The protocol hashes code, input matrices,
labels and provenance before any new training. Original frozen checkpoint/code
hashes are verified throughout. Never edit original run files to accommodate
this follow-up. There are no remote inference calls or automatic uploads.

Dependencies for feature extraction are isolated in `.venv_fe`; the established
MVP `.venv` is reused unchanged for official TabM/LightGBM training. See the two
environment lock snapshots. `raw`, `derived`, predictions and checkpoints stay
local. Upstream code redistribution permission has not been established.

Additional artifacts: `FEATURE_TRANSFER_REPORT_ZH.md` contains the complete
comparison; `FEATURE_TRANSFER_GUIDE_ZH.md` explains the transformations;
`feature_catalog.csv` maps the two champion-derived representations to uniquely
matched official field definitions without exposing customer records.
`aggregate_arithmetic_check.json` records 220 direct NumPy comparisons on a real
tax table. `verification.json` records metric recomputation and a cached `run`
that creates no additional fit. All these checks are by the same implementer;
there is no claim of an independent external review.
