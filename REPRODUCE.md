# Reproduction guide

This repository is a public source/evidence export of the 2026-10-14 experiments. Publication on 2026-10-15 did **not** resume training. The paused ExcelFormer checkpoint and all Home Credit records, row predictions and model files remain local. There are three different operations below: reading published evidence, performing a fresh experiment, and recovering the original paused job. They are not interchangeable.

## 1. Read and verify without data, authentication, or a GPU

From the cloned repository root, with Python:

```powershell
python scripts/verify_public_release.py --summary
```

This uses only the standard library. It checks the release hashes and aggregate-table consistency. It does not recompute metrics from individual predictions and does not claim independent replication. Read [progress](PROGRESS_ZH.md), [core report](evidence/core/reports/PORTFOLIO_REPORT_FINAL.md), [feature transfer](evidence/feature_transfer/FEATURE_TRANSFER_REPORT_ZH.md), and [paused full-recipe snapshot](evidence/full_solutions/reports/FULL_SOLUTIONS_REPORT_ZH.md).

Historical manifests under `evidence/` are archival evidence. Their internal hashes describe the original local artifacts, some of which are deliberately not public. Paths in exported evidence were redacted. `PUBLIC_EXPORT_PROVENANCE.json` maps original file hashes to public hashes; `RELEASE_MANIFEST.json` verifies this public package. Never edit an old freeze manifest to make a new run pass.

## 2. Recreate the core environment

The recorded machine ran Windows, Python 3.12.7, PyTorch 2.5.1+cu121 and an RTX 4080 SUPER with 16 GB VRAM. Four CPU threads were used. `requirements.lock` records installed versions, including the CUDA build. Reuse a compatible authorized environment or create a new venv; never upgrade global dependencies.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match -r requirements.lock
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
& ./.venv/Scripts/python.exe -m pytest -q
```

On Linux/macOS use `.venv/bin/python`. No GPU training is required for the synthetic contract tests. The real-engine integration test correctly **skips** in a fresh checkout without a locally executed AbsBox run; it must not be called a fresh integration PASS. Historical actual-engine evidence is separately published under `evidence/core/absbox/`.

## 3. Acquire only the authorized 2024 subset

Use your own Kaggle account and accept the competition's terms yourself if required. Neither this repository nor an automated helper accepts terms or supplies credentials. Never commit a token or OAuth credential file. With the pinned Kaggle SDK:

```powershell
& ./.venv/Scripts/kaggle.exe auth login
& ./.venv/Scripts/tabred.exe download homecredit-default --output-path data --max-workers 1
```

The source is **Home Credit — Credit Risk Model Stability 2024**, via the official TabReD downloader. Do not substitute the 2018 competition or download all TabReD datasets. The archive registry SHA256 is `fdcdcfe9b67e9ac8fae0291ee0aa8b492e41c123f99c9d3cecae2b78602c7f30`; extracted-file hashes and the exact A/B/C/D boundary metadata are in the data audit. A successful download command alone is not sufficient verification. Existing verified caches are reused.

## 4. Fresh core reproduction: explicit, separate run

This section starts a new experiment only when you deliberately execute it. These commands were **not executed during publication**. A full fresh experiment is distinct from replaying the historical checkpoint bytes.

```powershell
& ./.venv/Scripts/python.exe scripts/prepare_reproduction.py --run-id homecredit_reproduction_20261015
& ./.venv/Scripts/python.exe -m credit_mvp smoke --config configs/reproduction.yaml
& ./.venv/Scripts/python.exe -m credit_mvp run --config configs/reproduction.yaml
& ./.venv/Scripts/python.exe -m credit_mvp report --run-dir runs/homecredit_reproduction_20261015
```

The helper refuses to overwrite a config or existing run directory. `run` performs audit → A/B training/selection → C-only calibration → freeze → one D evaluation → portfolio analysis → report → bounded local synthetic AbsBox attempt. Original test D is already known in the published study: a rerun cannot be advertised as a newly unseen research holdout. GPU/kernel/version differences can change fitted weights and scores; fresh-run receipts retain their own identity.

The frozen configuration preserves three candidates and final seeds 42/0/1, AUC-based selection, positive-slope sigmoid calibration, fixed within-week retention rates and illustrative stress assumptions. No source-paper score is a pass threshold. `report` for a **locally completed** run reads its cache; it cannot operate on the public evidence directory as if withheld checkpoints/predictions were present.

## 5. Optional real local AbsBox engine

Synthetic requests and actual recorded responses are available without an engine. For a newly requested integration run, existing Docker/local permission is required:

```powershell
docker pull yellowbean/hastructure@sha256:aae12df53bb54f281dc93359297d617dd78c203ccd36c1f2c90e33b47f25438c
docker run -d --name home-credit-hastructure -p 127.0.0.1:8081:8081 yellowbean/hastructure@sha256:aae12df53bb54f281dc93359297d617dd78c203ccd36c1f2c90e33b47f25438c
& ./.venv/Scripts/python.exe -m credit_mvp absbox --run-dir runs/homecredit_reproduction_20261015
```

Use `docker start home-credit-hastructure` if that exact container already exists and is stopped. Do not start a duplicate engine. The code accepts localhost only and synthetic deal input only. Annual CDR/CPR are passed directly; recovery lag is a monthly period count. Home Credit probabilities are not converted to CDR. Failure remains `EXTENSION_BLOCKED`; synthetic hand calculations never substitute for engine responses.

## 6. Follow-up feature and full-recipe studies

The current follow-up is **PAUSED_BY_USER**. Its original local training/monitor processes were stopped; ExcelFormer completed three epochs and has a saved fourth-epoch position. S6E7 trained zero members. The following describes future explicit reproduction, not an instruction to restart the paused job automatically.

The scripts retain the historical folder names. Set the new core-run path before running them:

```powershell
$env:CREDIT_MVP_CORE_RUN = (Resolve-Path runs/homecredit_reproduction_20261015).Path
uv venv --python 3.12 experiments/champion_features_20261014/.venv_fe
uv pip install --python experiments/champion_features_20261014/.venv_fe/Scripts/python.exe -r experiments/champion_features_20261014/requirements-feature-lock.txt
& ./.venv/Scripts/python.exe scripts/restore_upstream.py --champion
& ./.venv/Scripts/python.exe experiments/champion_features_20261014/acquire.py
& ./experiments/champion_features_20261014/.venv_fe/Scripts/python.exe experiments/champion_features_20261014/features.py
& ./.venv/Scripts/python.exe experiments/champion_features_20261014/training.py prepare
& ./.venv/Scripts/python.exe experiments/champion_features_20261014/training.py baseline
& ./.venv/Scripts/python.exe experiments/champion_features_20261014/training.py run
& ./.venv/Scripts/python.exe experiments/champion_features_20261014/training.py report
```

The raw downloader requests only the recorded training Parquet files and feature definitions, with existing-cache hash checks. Raw tables, derived matrices and row outputs stay ignored. The source-restoration helper uses recorded Kaggle versions and SHA256 values; if access or byte identity fails it stops. It never uses a mirror, substitutes a newer notebook, retrieves the author's private model files, or accepts new terms. It was tested locally against the existing authorized source cache during publication; remote historical-version availability is not guaranteed.

The full-recipe wrappers are in `experiments/full_solutions_20261014/`. Runtime package pins and the complete observed package-version inventory are provided. Historical `.venv_full` reused system packages read-only with pinned local overrides; a fresh resolved full environment was **not** claimed or tested during publication. Optional packages include `catboost==1.2.8`, `lightautoml==0.3.8`, `masamlp==0.3.0`, `catstat==0.4.0`, `category-encoders==2.6.3`, `rtdl-revisiting-models==0.0.2`, and `rtdl-num-embeddings==0.0.12`. See [environment provenance](evidence/full_solutions/environment_versions.json). Validate their imports and synthetic tests before any future full run.

For ExcelFormer, restore the exact official implementation into ignored local directories using `python scripts/restore_upstream.py --excel`; byte hashes of the original core and constructed helper must match. Upstream source with unestablished redistribution rights is not bundled. Then, in a compatible isolated environment and **only for explicitly requested new work**:

```powershell
Set-Location experiments/full_solutions_20261014
python -m pytest -q test_transfer.py
python campaign.py prepare
python smoke.py
python campaign.py freeze
python campaign.py run --family champion
# Expensive, explicitly selected work; these were NOT resumed for publication:
# python campaign.py run --family excel
# python campaign.py run --family s6e7
python report.py
```

The champion command reproduces the documented **reconstruction**, not the inaccessible exact champion. Fresh full runs need the source provenance files listed in `PROTOCOL_LOCK.json`; archived locks belong to the historical experiment and must not be copied into a new run. The full script's family order, 36-hour wall cap, per-family caps, fixed 7×4×16 S6E7 structure, and no-new-C/D rule remain documented in `config.json`.

Exact recovery of the original paused ExcelFormer requires its local checkpoint, preprocessing, optimizer/RNG state and unchanged historical code. These are not distributed. It requires an explicit resume decision; elapsed budgets are not automatically reset. This public export cannot masquerade as that checkpoint recovery.

## 7. Official-formula analysis

`analyses/official_metric_20261015/` contains the completed **post-hoc** Gini stability analysis. The metric was added after the original results were known. All three fixed D seeds and all 13 completed B rows are included; no model was trained or reselected. The score is not a Kaggle submission or a leaderboard-rank estimate.

The public package includes the own metric implementation, synthetic tests, aggregate results, original `WEEK_NUM` audit, figure, protocol and result receipt. It contains no row predictions, labels or mapping records. First verify the package using only Python's standard library:

```powershell
python scripts/verify_public_release.py --summary
```

This also recalculates the official formula from the published weekly Gini values and checks the seed summaries and raw/calibrated equality. This verifies aggregate arithmetic; it does not independently reproduce weekly AUC from unavailable individual records.

With the pinned core environment, run the ten synthetic metric tests:

```powershell
& ./.venv/Scripts/python.exe -m pytest -q -o addopts= analyses/official_metric_20261015/test_metric.py
```

The archived `analyze.py` is the exact script used with the original authorized local workspace. Its `--project` option locates that workspace, but it still requires the historical cached row predictions, raw base table, metadata, split indices, original receipt paths and paused checkpoint. Those bytes are intentionally excluded from this repository. For an owner of the unchanged original workspace:

```powershell
& ./.venv/Scripts/python.exe analyses/official_metric_20261015/analyze.py --project '<original-authorized-workspace>'
```

With the existing result receipt, this checks hashes and exits `CACHE_ONLY`; it does not rescore or fit. The exact historical Windows receipts contain original path conventions, so this is not a portable fresh-run entry point. A public clone alone cannot execute this row-data check. Never remove receipts or relax hash checks to substitute a fresh run for the historical analysis. Use the aggregate verifier and synthetic tests for a data-free check.

The original core report used calendar weeks starting on Monday. This additional analysis uses recovered official `WEEK_NUM` (D: 69–91, B: 52–60), retains partial boundary weeks, and documents their small-sample limitation. The original frozen outputs remain archival and unchanged.

## Evidence and scope

Core D results and subsequent reused-B results are shown separately. Feature engineering and model selection on B are development evidence, not new independent validation. Unknown target horizon/maturity, inherited global unlabeled preprocessing, application rather than verified customer identity, normalized EAD=1 and assumed LGD remain limitations. See [DATA_CARD.md](DATA_CARD.md), [THIRD_PARTY.md](THIRD_PARTY.md), and [export boundaries](PUBLICATION_NOTES.md).
