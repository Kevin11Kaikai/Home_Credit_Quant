# Full requested solutions: evidence and execution

This independent follow-up preserves the completed MVP and feature-transfer experiments. It loads **A/B only** (267,645 and 40,596 records) using their existing immutable mapping. B is reused development data, not a new test set. C/D are never supplied to training or reporting functions.

The exact Home Credit champion remains `BLOCKED_DEPENDENCY`: the three original artifact references are unavailable, and the final feature masks and training configurations are not published in the inference notebooks. `CHAMPION_REPRODUCTION_AUDIT.json` records the actual checks. The executable substitute is explicitly a **three-family reconstruction**, not the exact champion. It includes five grouped A folds per family, LightGBM/DenseLight/CatBoost, published 0.40/0.24/0.36 blending, and the published 0.03 earlier-week score heuristic. The latter is a competition score adjustment, not calibrated PD.

The requested fraud repository contributes its **primary ExcelFormer pipeline**: numeric quantile scaling, missingness indicators, five chronological blocks of categorical target encoding, training-only MI ordering, MI-weighted feature mixing BCE, and a fresh initial fit plus continuation. Home Credit has no verified customer UID/card/address fields, so those fraud-specific feature families cannot be copied. Existing per-application credit history features are used; the limitation remains explicit. No prior fraud weights, data, scores, class weighting, resampling or distillation are imported.

The S6E7 transfer retains **7 folds × 4 members × 16 epochs**, FT-Transformer with PLR-lite, outer-fold-only exact-value target encoding with internal cross-fitting, cosine LR and equal averaging. Binary TE and binary BCE adapt the three-class source to this task. Batch 4096 is preserved with microbatch accumulation. The source prior correction is retained as a separate classification-only diagnostic; it never alters the event probability used for log loss/Brier.

All new fits are fixed before execution; there is no search for a Transformer win. `config.json` declares a new bounded follow-up budget (36 hours wall, separate per-family caps) without modifying older ledgers. Full-width A smoke results provide estimates, not completion claims. Every completed artifact has hashes and checkpoint reload verification. Neural mid-epoch recovery preserves optimizer, order, RNG and counters. `acceptance.json` reports actual completion, including a partial ensemble when fewer than 28 members finish; a partial ensemble is never called the full S6E7 recipe.

Run from this directory in PowerShell:

```powershell
& .\.venv_full\Scripts\python.exe -m pytest -q test_transfer.py
& .\.venv_full\Scripts\python.exe campaign.py prepare
& .\.venv_full\Scripts\python.exe smoke.py
& .\.venv_full\Scripts\python.exe audit_sources.py
& .\.venv_full\Scripts\python.exe campaign.py freeze
& .\.venv_full\Scripts\python.exe campaign.py run --family all
& .\.venv_full\Scripts\python.exe report.py
```

`run --family champion`, `excel` or `s6e7` provides scoped recovery. Completed artifact receipts are verified and reused. Do not run two campaign processes concurrently. A frozen-code change requires a documented repair record and a new protocol version; do not delete the lock to hide a change. The report command only reads cached predictions and never retrains or selects new configurations.

The initial preparation, source audit, one allowed smoke test, freeze, and unit tests have already been executed for this campaign. For recovery, verify the current process has ended, then use the scoped `run` command; do not repeat completed preparation or the smoke test. Existing family and wall-clock budgets continue across recovery.

`REPRODUCTION_BOUNDARIES.md` maps each source recipe to its implemented components and explicit adaptations. `environment_snapshot.json` records the complete visible distribution list and hardware; the small `requirements.lock.json` lists protocol-critical packages. The environment reuses existing system packages read-only via an independent virtual environment with local pinned overrides. It is not a self-contained offline wheel bundle.

`monitor.py --pid <training_pid>` is a cache-only helper for this one running job. It refreshes the report when completed receipts or failure files change, and exits when that exact process ends. It never retries training, uploads artifacts, or starts additional experiments. Check `campaign_state.json`, `runs/*/progress.json`, `budget.json`, and `acceptance.json` together when inspecting a running job.

