# 中文交付摘要

核心：CORE_COMPLETE；扩展：EXTENSION_COMPLETE。

真实主结果：

| model | variant | calibration_status | roc_auc | average_precision | log_loss | brier | mean_probability | observed_event_rate | probability_bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | raw | FITTED | 0.862234 | 0.166495 | 0.085750 | 0.019990 | 0.033137 | 0.021946 | 0.011191 |
| LightGBM | calibrated | FITTED | 0.862234 | 0.166495 | 0.087655 | 0.020202 | 0.037305 | 0.021946 | 0.015359 |
| TabM | raw | FITTED | 0.863624 | 0.170869 | 0.086195 | 0.019971 | 0.034256 | 0.021946 | 0.012310 |
| TabM | calibrated | FITTED | 0.863624 | 0.170869 | 0.086383 | 0.020013 | 0.034845 | 0.021946 | 0.012899 |
| Constant | raw | NOT_APPLICABLE | 0.500000 | 0.021946 | 0.106961 | 0.021542 | 0.030753 | 0.021946 | 0.008807 |

LightGBM: calibrated minus raw log loss = +0.001905; Brier = +0.000212. These are observed probability-score changes; inspect reliability and bias before claiming better calibration. TabM: calibrated minus raw log loss = +0.000189; Brier = +0.000042. These are observed probability-score changes; inspect reliability and bias before claiming better calibration.

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

上游存在全量无标签预处理统计；标签期限、成熟时间、客户独立性均未核验。归一化损失是事件标签乘假设 LGD，不是实际金额损失。
