# Sources, attribution and distribution scope

This file accompanies an explicitly authorized public export. The statements below about historical source collection describe the original experiment. The only upstream source files bundled here are license texts; implementation libraries are installed from pinned packages, and optional notebook/ExcelFormer sources must be restored locally from their official source with hash checks. Raw competition data and private model files are excluded.

- Home Credit 2024 champion: **yuuniee / yuuniekiri**, official first-place writeup and three inference notebooks. See `evidence/full_solutions/SOURCE_MANIFEST.json` and `SOURCE_AUDIT_ADDENDUM.json`. The displayed DenseLight v11 notebook reports Apache 2.0, while its input includes a private dataset. The downloaded v12 and the other notebooks are referenced rather than redistributed. Exact model files/masks remain unavailable.
- Requested fraud workflow: **Kevin11Kaikai/Fraud_Detection_Tabular_Transformer**, commit `d10eb6315319b20d0be7a94e63a4f89c6381d1be`. No fraud data, weights or metrics are redistributed. Its primary ExcelFormer/ordered-TE/MI-feature-mix recipe motivated the explicitly adapted credit implementation.
- ExcelFormer: **WhatAShot/ExcelFormer**, commit `17f70526390e70390bb8c8ec3850697eb730f9cd`. Original core/helper source is not bundled because a redistribution license was not established. The optional restoration script retrieves only the pinned official files into ignored local directories and checks the recorded hashes.
- S6E7: fourth-place writeup by **Ricky / rickyds97**, crediting **Masaya Kawamata** and notebook `masayakawamata/s6e7-ft-transformer-v2-cv-0-95063`, v1. Packages `masamlp==0.3.0` and `catstat==0.4.0` are imported from their released implementations. The reported source balanced accuracy is not a credit-model result.
- DenseLight uses official **LightAutoML 0.3.8**; CatBoost uses **1.2.8**. Their implementations are not vendored. Their distributions and transitive dependencies retain their own notices.

## Original core source record

# Sources, licenses and reuse

This independent project uses no Anoris Capital/JPMorgan private data, models or internal systems. See `SOURCE_MANIFEST.json` for exact URLs and SHA256 values, and `upstream/` for preserved license texts (implementation snapshots are not distributed here). Repository code licenses do **not** automatically license Home Credit data.

| Component | Fixed source/version | License / credit | Reuse |
|---|---|---|---|
| TabReD | `b5ef15b3749f30da7a1eb8fba21a5b54d706bf32`, package 0.1.2 | Apache-2.0; Ivan Rubachev, Nikolay Kartashev, Yury Gorishniy, Artem Babenko | Official downloader/registry; inspect preprocessing and metadata conventions; do not reimplement raw feature engineering |
| TabM | `28e47ae301c92ec37787dde1ce923a0793f405b4`, 0.0.3 | Apache-2.0; Yury Gorishniy / Yandex LLC | Official installed `tabm.py`; example.ipynb consulted for architecture, head training and probability averaging; no test callback reused |
| rtdl-num-embeddings | `a8fc25025c83f2321c63ff127a3bcef83bb1bfb5`, 0.0.12 | Actual **package/LICENSE** is Apache-2.0, Yandex LLC; repository root license differs | Official compute_bins / PiecewiseLinearEmbeddings version B |
| LightGBM | 4.6.0 | MIT; Microsoft and contributors | Native CPU binary booster/categorical support; official parameter documentation |
| scikit-learn | 1.5.1 | BSD-3-Clause; scikit-learn developers | Probability metrics; official calibration concepts. Positive-slope two-parameter scipy optimization is local code |
| AbsBox | `1a08f8a220bf68f53677271dc2d961c53916d2e8`, 0.52.8 | Apache-2.0; yellowbean and contributors | Official `absbox/examples/baseCase.py:test01`, client API and analytics docs |
| Hastructure | image `yellowbean/hastructure@sha256:aae12df53bb54f281dc93359297d617dd78c203ccd36c1f2c90e33b47f25438c`, runtime 0.52.8 | Preserved BSD-style/custom engine LICENSE; GitHub SPDX NOASSERTION | Local container, no mounts, localhost only. Inspected source HEAD is not claimed to prove image build commit |

Official entry points: [TabReD](https://github.com/yandex-research/tabred), [Home Credit preprocessing](https://github.com/yandex-research/tabred/blob/b5ef15b3749f30da7a1eb8fba21a5b54d706bf32/preprocessing/homecredit.py), [TabM](https://github.com/yandex-research/tabm), [TabM example](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb), [LightGBM parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html), [scikit-learn calibration](https://scikit-learn.org/1.5/modules/calibration.html), [AbsBox](https://github.com/yellowbean/AbsBox), [AbsBox installation](https://absbox-doc.readthedocs.io/en/latest/installation.html).

The initially requested stable documentation pages were retrieved and byte-hashed; the live pages currently identify LightGBM 4.7.0 and scikit-learn 1.9.1, while this environment deliberately uses the explicitly pinned versions above. Version-specific pages and installed APIs are the implementation authority. No published Home Credit test score or data-specific tuned configuration informed the search.

## Data access

Only `irubachev/tabred` → `preprocessed/homecredit-default.tabred` was requested. Its registry SHA256 is `fdcdcfe9b67e9ac8fae0291ee0aa8b492e41c123f99c9d3cecae2b78602c7f30`. The source is [Home Credit — Credit Risk Model Stability 2024](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability), governed by its competition/data-use terms. The official command is `tabred download homecredit-default --output-path data --max-workers 1`. The initial authentication block was resolved by the user completing official OAuth after accepting the competition rules. On 2026-10-14 the official downloader verified competition access, checked the registry archive hash, and extracted the single requested subset. Its temporary archive was removed by upstream cleanup; extracted file hashes are preserved in the actual data audit. No terms were automatically accepted, and no raw rows or labels are published.

Upstream full-data missing-rate, category-frequency and constant-feature processing is documented as inherited/transductive preprocessing. Rare categories cannot be reconstructed. Label horizon, maturity, customer identity and genuine EAD/LGD remain unverified. The inspected preprocessing source and archive naming conventions differ; exact artifact-generation lineage is not independently established.

## Attribution

Rubachev, Kartashev, Gorishniy and Babenko. *TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks*. ICLR 2025. [OpenReview](https://openreview.net/forum?id=L14sqcrUC3).

Gorishniy and collaborators. *TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling*. ICLR 2025. Preserve the citation and license distributed in the official TabM repository.

All further transitive dependencies and exact installed versions are listed in `requirements.lock`; their distributions retain their own notices. Original local snapshots were unmodified except filenames introduced by the source archiver; they are not included in this public export. No copyright or license header has been removed.
