# Verified visual comparison

Scope: successfully completed champion-feature adaptation and transfer;
the complete champion fitted models, final feature masks and ensemble were
not reproduced. All plots use the same 40,596 B development records, seed 42,
and raw probabilities. B was reused for checkpoint selection. No independent
test claim or statistical significance is established.

- `four_metric_comparison`: absolute ROC-AUC, AP, log loss and Brier for both models and all four input representations. Point plots use explicitly labelled local axis ranges.
- `changes_vs_original`: each model compared with its own original representation. Cell values are signed raw-score differences; color denotes favourable or unfavourable direction only, not magnitude or statistical significance.
- Each figure is available as PNG, SVG and PDF.
- `plotted_differences.csv` contains the plotted deltas; `plot_receipt.json` records source and figure hashes.

Reproduce from `${SOURCE_WORKSPACE}`:

```powershell
& '.venv\Scripts\python.exe' 'experiments\champion_features_20261014\plot_comparison.py'
```

This command verifies cached result hashes and preserves existing artifacts.
It does not train models, generate predictions or read individual labels.
