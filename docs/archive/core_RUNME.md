# Run and resume

Run from `${SOURCE_WORKSPACE}` in PowerShell. The independent `.venv` is already installed and verified. Python is 3.12.7; PyTorch is 2.5.1+cu121 and CUDA is available on the actual RTX 4080 SUPER.

```powershell
Set-Location ${SOURCE_WORKSPACE}
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe -m credit_mvp smoke --config configs\mvp.yaml
& .\.venv\Scripts\python.exe -m credit_mvp run --config configs\mvp.yaml
& .\.venv\Scripts\python.exe -m credit_mvp report --run-dir runs\homecredit_temporal_v1_20261014
& .\.venv\Scripts\python.exe -m pytest -q
```

All four interfaces were actually exercised. `run` returns exit code **2** for CORE_PARTIAL, and 0 only for CORE_COMPLETE. The downloader's exit code alone is never treated as proof of acquisition. `smoke` is cached after the single brief fit per model; in this delivery it explicitly reports `SYNTHETIC_ENGINEERING_ONLY_REAL_DATA_UNAVAILABLE`. Existing valid smoke weights are not retrained merely to refresh a source-code hash.

After a completed formal evaluation, these local commands verify saved arithmetic and compose the portfolio-facing material from the same cache:

```powershell
& .\.venv\Scripts\python.exe scripts\audit_final_outputs.py
& .\.venv\Scripts\python.exe scripts\finalize_delivery.py
& .\.venv\Scripts\python.exe scripts\render_reliability_panels.py
& .\.venv\Scripts\python.exe scripts\write_portfolio_materials.py
```

They do not fit a model, choose a seed, create new predictions or add another formal test-evaluation record. `PORTFOLIO_REPORT.md` is the pipeline's technical report; `PORTFOLIO_REPORT_FINAL.md` is its separate portfolio-facing narrative, with `SUMMARY_FINAL_ZH.md` and `RESUME_BULLETS_FINAL.md`. The arithmetic audit explicitly distinguishes same-agent recomputation from independent external validation. Previously completed delivery receipts are preserved in the run history.

## Official data acquisition and authentication

The user accepted the [2024 competition rules](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/rules) and completed the official OAuth flow on 2026-10-14. The downloader subsequently verified competition access and acquired only the requested subset. Existing data are reused; no login or download is needed merely to read reports. On a fresh machine, complete the following official browser flow using an authorized Kaggle account. Do not paste a token into chat or commit it. The agent has not accepted any terms.

```powershell
& .\.venv\Scripts\kaggle.exe auth login
```

The installed Kaggle SDK's OAuth flow stores credentials at `%USERPROFILE%\.kaggle\credentials.json`; its OAuth credential store does not follow `KAGGLE_CONFIG_DIR`. This is distinct from legacy API-token configuration. The experiment never reads or logs credential values.

Then the package invokes this exact official command (one subset only):

```powershell
& .\.venv\Scripts\tabred.exe download homecredit-default --output-path data --max-workers 1
```

`KAGGLE_CONFIG_DIR` may point to an authorized credential directory. If it is unset, the standard Kaggle location is used; when no such directory exists the program uses the ignored project `.kaggle` directory. No secret value is written to experiment logs. Every explicit `run` makes at most one downloader invocation, with a 600-second outer timeout; the upstream downloader has its own bounded retry policy. No unofficial mirror, full eight-dataset download, `--force`, or raw multi-table rebuild is used.

The original authentication-blocked diagnostic is preserved in the run history. The actual immutable audit records 381,664 rows, 696 base features, and complete-day B/C boundaries: A is 2019, B is January-February 2020, C is March-April 2020, and D is 2020-05-01 through 2020-10-05. The current archive uses `info.json` with nested `task.type`; its official directory name and hash-checked downloader establish dataset identity. The parser adaptation was recorded before any formal training or D access. A missing/ambiguous schema or temporal conflict still blocks experiments rather than being guessed.

## Frozen experiment and recovery

The YAML fixes all candidates, seeds, metrics, calibration bounds, q values, random seeds and stress assumptions. A `PROTOCOL_LOCK.json` is created before the first formal fit. `FROZEN_MANIFEST.json` is created after selected checkpoints, calibration and C-based reliability edges exist. D cannot enter training or calibration. The first test-access receipt permanently records D as seen, including an interrupted evaluation. Cached prediction and evaluation receipts prevent silent replacement; `report` never reselects a model or reads raw labels.

Budget state persists by model family **and individual trial**; a resumed TabM fit receives only its unused cap. TabM checkpoints preserve optimizer, preprocessing references, minibatch position, permutation and RNG states. Native LightGBM snapshots are retained after interruption. Because its public serialized model does not expose all internal RNG state, an interrupted native fit is explicitly retained as partial rather than silently restarted and claimed bitwise equivalent. Time-limited fits are never called converged. No previous run or evidence directory is deleted.

If code, configuration or a frozen artifact must change after D has been accessed, preserve the current run and document a new version plus a correction record; the previously observed D is not a fresh holdout. Data changes fail hash verification. To publish code later, exclude `.venv`, `.kaggle`, raw `data`, row-level Parquet and model artifacts. Public release still requires a separate explicit action.

## Local synthetic AbsBox

The real demonstration is complete and cached. It used official `absbox.examples.baseCase.test01`, AbsBox 0.52.8 and engine 0.52.8, with no real customer rows, no mounts and localhost binding only. To recreate the local engine if needed:

```powershell
docker pull yellowbean/hastructure@sha256:aae12df53bb54f281dc93359297d617dd78c203ccd36c1f2c90e33b47f25438c
docker run -d --name anoris-credit-hastructure -p 127.0.0.1:8081:8081 yellowbean/hastructure@sha256:aae12df53bb54f281dc93359297d617dd78c203ccd36c1f2c90e33b47f25438c
```

If that named container already exists and is stopped, use `docker start anoris-credit-hastructure` instead. There is no need to recreate it just to read completed results. The `absbox` command verifies and reuses cached responses once complete:

```powershell
& .\.venv\Scripts\python.exe -m credit_mvp absbox --run-dir runs\homecredit_temporal_v1_20261014
& .\.venv\Scripts\python.exe scripts\audit_absbox_outputs.py
```

The extension guard accepts localhost and the explicit synthetic deal only. CDR/CPR are annual inputs accepted directly by this API; monthly recovery lags are period counts for Monthly assets. Up to three bounded attempts and 30 active minutes are allowed. The completed integration test checks actual saved response hashes and reconciliation, not a mocked engine. The secondary Decimal audit discloses small engine loss-decomposition residuals and the junior tranche's residual-cash semantics.

## Rebuild the environment

Only if the local environment is missing:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match -r requirements.lock
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
```

`requirements.lock` fixes every installed package version and the CUDA build. `SOURCE_MANIFEST.json` additionally pins implementation bytes and upstream commits. The old psutil 5.9 build was found incompatible with this Python 3.12 runtime and replaced inside `.venv` by 7.0.0 before formal training. No global package was upgraded.
