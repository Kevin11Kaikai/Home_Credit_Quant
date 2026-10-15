# Credit Risk under Temporal Shift: Calibrated Default Prediction and Portfolio Stress Analysis

## Question and actual status

This independent public-data portfolio project targets credit-event prediction, chronological validation, probability calibration, normalized loan-pool analysis and Python research engineering. It uses no private Anoris Capital or JPMorgan data, systems, weights or performance evidence. The empirical questions are deliberately limited: under a fixed temporal split and bounded search, how do LightGBM and official TabM compare in discrimination, probability quality and stability; and does a separately held-out positive-slope sigmoid calibrator improve later probability scores and pool loss estimates?

The core status is **CORE_COMPLETE**. The extension status is **EXTENSION_COMPLETE**. Formal empirical results available: **True**. Matched evaluated seeds: **3**, [0, 1, 42]. Missing items: none. Results below describe executed work only. Synthetic contract checks are software evidence, not measured credit-risk performance. A blocked download does not establish a model failure or a negative empirical answer.

## Sources and data limitations

The intended dataset is TabReD homecredit-default, derived from Home Credit — Credit Risk Model Stability 2024, not the 2018 competition. TabReD is pinned to `b5ef15b3749f30da7a1eb8fba21a5b54d706bf32` and TabM to `28e47ae301c92ec37787dde1ce923a0793f405b4`. SOURCE_MANIFEST.json records retrieved official files, hashes, licenses and the exact single-subset acquisition command. The expected archive hash is independently recorded from the upstream registry. Kaggle authentication and competition access remain the user's responsibility; this project does not accept new terms or use alternative mirrors. Raw records and local predictions are excluded from public publication.

The inspected upstream script computes missingness and category cardinalities on the full feature collection before temporal splitting. It removes high-missingness and constant columns, filters some high-cardinality strings, and fits OrdinalEncoder with min_frequency=0.01 globally. It also uses full-table extrema for dtype downcasting. These are inherited/transductive preprocessing limitations. They are not, by themselves, demonstrated target-label leakage. Nevertheless, strict point-in-time preprocessing cannot be claimed. Refitting a vocabulary on A cannot undo rare categories already merged upstream. The source and current archive use different documented filenames, so exact artifact-generation lineage also remains unverified until inspected conservatively.

The source describes a default target, but an exact observation horizon and label maturity schedule have not been verified. Outputs are therefore dataset target-event probabilities or PD proxies, not one-year PDs. The documented metadata are date_decision, encoded as Unix epoch days, and index_in_full, a source row position. The latter is not a customer identifier; customer-level independence cannot be claimed. Exposures, realized recovery, default timing and contractual cashflows are not established by anonymous amount columns. These limitations prevent interpretation as a tradeable historical backtest.

## Temporal protocol and implementation

| partition | n | start | end | unique_dates |
| --- | --- | --- | --- | --- |
| A | 267645 | 2019-01-01 | 2019-12-31 | 365 |
| B | 40596 | 2020-01-01 | 2020-02-29 | 60 |
| C | 17422 | 2020-03-01 | 2020-04-30 | 61 |
| D | 56001 | 2020-05-01 | 2020-10-05 | 157 |

The protocol keeps official train as A and official final test as D. The distinct validation dates are sorted and split at their midpoint: earlier complete dates form B and later complete dates form C. Boundaries depend only on metadata, never event rates. Strict A-before-B-before-C-before-D ordering and pairwise disjoint row indices are mandatory. Conflicting dates block evaluation instead of triggering a random replacement. This is a documented adaptation of TabReD, not an exact reproduction of its paper scores.

All newly fitted medians, constant masks, scaling, categorical vocabularies and numerical bins use A only. Both models receive the same underlying feature information. Metadata and labels cannot enter feature matrices. Unknown categories have an explicit code; missing categories use a separate code. LightGBM uses CPU native categorical features. TabM uses the official k=32 architecture, three width-256 blocks and 32-bin, dimension-16 piecewise-linear numerical embeddings. Each head contributes its own binary cross-entropy; inference averages sigmoid probabilities. No weighting, resampling or external ensemble is used.

Three frozen candidates per family use seed 42 and B AUC selection, with log loss and candidate ID resolving exact ties. The first candidate of each model precedes further searches. Additional fixed-configuration seeds follow 0 then 1, without seed selection. Limits are 20 minutes per LightGBM fit and two hours cumulatively, and 45 minutes per TabM fit and four hours cumulatively; interrupted results retain their actual status. The verified machine is NVIDIA GeForce RTX 4080 SUPER, with 31.72 GiB RAM. CPU threads are limited to 4. JSON/CSV tracking is local because this empty workspace had no MLflow configuration.

## Controlled comparison

| model | variant | calibration_status | roc_auc | average_precision | log_loss | brier | mean_probability | observed_event_rate | probability_bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | raw | FITTED | 0.862234 | 0.166495 | 0.085750 | 0.019990 | 0.033137 | 0.021946 | 0.011191 |
| LightGBM | calibrated | FITTED | 0.862234 | 0.166495 | 0.087655 | 0.020202 | 0.037305 | 0.021946 | 0.015359 |
| TabM | raw | FITTED | 0.863624 | 0.170869 | 0.086195 | 0.019971 | 0.034256 | 0.021946 | 0.012310 |
| TabM | calibrated | FITTED | 0.863624 | 0.170869 | 0.086383 | 0.020013 | 0.034845 | 0.021946 | 0.012899 |
| Constant | raw | NOT_APPLICABLE | 0.500000 | 0.021946 | 0.106961 | 0.021542 | 0.030753 | 0.021946 | 0.008807 |

The constant reference is mean(y_A). Average precision is named explicitly and is not relabeled as generic PR-AUC. Weekly tables retain small and poor-performing weeks, record sample counts and mark single-class AUC as NA. When enough consecutive real calendar weeks exist, 200 paired moving-block resamples use two-week blocks shared by both fixed models. Their intervals address short-range temporal sampling variation, not retraining uncertainty, unidentified customers or arbitrary longer dependencies. Seed means and standard deviations are reported only over matched seeds and do not replace data uncertainty.

## Independent calibration

LightGBM: calibrated minus raw log loss = +0.001905; Brier = +0.000212. These are observed probability-score changes; inspect reliability and bias before claiming better calibration. TabM: calibrated minus raw log loss = +0.000189; Brier = +0.000042. These are observed probability-score changes; inspect reliability and bias before claiming better calibration.

The only calibrator is sigmoid(exp(theta) * logit(clipped raw probability) + b), optimized by unweighted binary log loss on C. Its identity initialization, bounds and iteration cap are frozen. Single-class or failed fits remain explicitly failed and return raw probabilities. C fitting scores are not presented as generalization results. C predictions also define deduplicated reliability-bin edges covering [0,1]; D labels never tune bins. Since positive slope preserves ranking mathematically, calibration cannot create new discrimination or improve a fixed-count pool's actual event rate. Floating-point clipping and saturation can introduce ties, so pool masks are always generated from raw probability and stable row ID and reused exactly.

## Loan-pool interpretation

| model | variant | n | exposure | actual_retention | observed_event_rate | normalized_EL | normalized_observed_proxy | error_per_exposure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | raw | 39189 | 39189.000000 | 0.699791 | 0.004466 | 0.004002 | 0.001786 | 0.002216 |
| LightGBM | calibrated | 39189 | 39189.000000 | 0.699791 | 0.004466 | 0.004852 | 0.001786 | 0.003066 |
| TabM | raw | 39189 | 39189.000000 | 0.699791 | 0.004721 | 0.003856 | 0.001888 | 0.001967 |
| TabM | calibrated | 39189 | 39189.000000 | 0.699791 | 0.004721 | 0.003822 | 0.001888 | 0.001934 |

Retention fractions are fixed at 0.50, 0.70, 0.90 and 1.00, with 0.70 designated in advance. Selection occurs within each test week, retaining floor(q times the weekly count). This gives equal exposure across models; rounding is logged, including empty selected batches. Twenty fixed random seeds provide same-week, same-count controls. This is batch selection, not a sequential approval policy. Calibration changes estimated loss, not the records chosen. Smaller total losses from smaller pools are not interpreted as improvement; comparisons use loss per unit exposure.

Each record contributes EAD=1 normalized unit. Expected proxy loss sums probability times assumed LGD; the observed proxy sums event labels times the same assumed LGD. Neither quantity is an actual dollar loss or realized net credit loss. Base, Adverse and Severe use fixed odds multipliers 1.0/1.5/2.0 and LGDs 0.40/0.60/0.80 on unchanged pools. These illustrative parameters are not estimated macroeconomic or regulatory stresses. No stress-period accuracy, investment return, Sharpe ratio, VaR, expected shortfall or default-correlation model is claimed. Formula-driven loss increases are not independent research discoveries.

## AbsBox extension

Status: **EXTENSION_COMPLETE**. Real engine response obtained: **True**.

| scenario | tranche | principal_repaid | ending_balance | max_interest_due | ending_interest_due |
| --- | --- | --- | --- | --- | --- |
| ZeroPressure | A1 | 1000.000000 | 0.000000 | 0.000000 | 0.000000 |
| ZeroPressure | B | 1000.000000 | 0.000000 | 0.000000 | 0.000000 |
| Base | A1 | 1000.000000 | 0.000000 | 0.000000 | 0.000000 |
| Base | B | 1000.000000 | 0.000000 | 0.000000 | 0.000000 |
| Adverse | A1 | 1000.000000 | 0.000000 | 0.000000 | 0.000000 |
| Adverse | B | 1000.000000 | 0.000000 | 0.000000 | 0.000000 |
| Severe | A1 | 1000.000000 | 0.000000 | 0.000000 | 0.000000 |
| Severe | B | 1000.000000 | 0.000000 | 0.000000 | 0.000000 |

Independent Decimal recomputation confirmed cash residuals of 0.00 in all four cases. The engine reports Default minus Recovery minus Loss residuals of 0.15, 0.15 and 0.12 under Base, Adverse and Severe; their cause is unverified and the raw values are preserved. The junior B interest field includes equity residual distributions and released overcollateralization, rather than pure contractual interest.

The extension uses only the official independent synthetic senior/junior example. Home Credit probabilities are never converted into annual CDR. Fixed Base/Adverse/Severe annual CDRs are 2%/5%/10%, annual CPRs 5%/10%/15%, recovery rates 60%/40%/20%, and monthly recovery lags 3/6/9. The API accepts annual rates directly. Only localhost is allowed. Requests, raw responses, engine/client versions and cashflows are retained when execution succeeds; a zero-pressure case checks initial collateral, liabilities and cash conservation. Unpaid interest balances are not repeatedly summed as new shortfalls. This demonstrates waterfall mechanics, not a complete CLO valuation or investment strategy.

## Reproducibility and limits

The package provides smoke, run and report commands plus automated contracts. A pretraining lock fixes decisions; a later frozen manifest hashes code, splits, preprocessing, checkpoints, calibrators and evaluation settings. A test-access receipt precedes reading D labels. One completion record belongs to one protocol; reporting reads cached artifacts and verifies their hashes. An observed D can never become unseen again after a bug fix. All factual numerical claims require the corresponding local artifacts. Authentication blockage is preserved as CORE_PARTIAL, while the independent extension can succeed or remain blocked without inventing missing predictive results.
