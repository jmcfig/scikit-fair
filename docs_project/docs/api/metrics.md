# skfair.metrics

```python
from skfair.metrics import <function_name>
```

Fairness metrics are exposed as **counterpart pairs**: each base measure has
both a *difference* form (suffix `_difference`, ideal = 0) and a *ratio* form
(suffix `_ratio`, ideal = 1). Canonical names are the full descriptive forms;
short aliases (`spd`, `di`, `eod`, `eor`, `aod`, `aor`, `fpr_diff`, `ppv_ratio`,
...) are also exported.

All metric metadata — function, family, ideal value, aliases — lives in a single
registry, `skfair.metrics.REGISTRY`. The `benchmark` helper and the fairness-only
`METRICS` view are derived from it.

---

## Registry & benchmark

::: skfair.metrics.benchmark

`METRICS` maps every canonical fairness-metric name to a `MetricSpec`
(`func`, `kind`, `family`, `ideal`, `display`, `aliases`). `DEFAULT_BENCHMARK_METRICS`
is the one-per-family subset evaluated when `benchmark` is called without
`metrics`. `REGISTRY` is the full registry including performance metrics.

---

## Fairness metrics

### Independence (depends on Ŷ only)

::: skfair.metrics.statistical_parity_difference

---

::: skfair.metrics.disparate_impact

---

### Separation — True Positive Rate

::: skfair.metrics.true_positive_rate_difference

`eod`, `tpr_diff` and `equal_opportunity_difference` are aliases.

---

::: skfair.metrics.true_positive_rate_ratio

`eor`, `tpr_ratio` and `equal_opportunity_ratio` are aliases.

---

### Separation — False Positive Rate

::: skfair.metrics.false_positive_rate_difference

---

::: skfair.metrics.false_positive_rate_ratio

---

### Separation — True Negative Rate

::: skfair.metrics.true_negative_rate_difference

---

::: skfair.metrics.true_negative_rate_ratio

---

### Separation — False Negative Rate

::: skfair.metrics.false_negative_rate_difference

---

::: skfair.metrics.false_negative_rate_ratio

---

### Separation — Combined odds (FPR + TPR)

::: skfair.metrics.average_odds_difference

---

::: skfair.metrics.average_odds_ratio

---

### Sufficiency — Positive Predictive Value (predictive parity)

::: skfair.metrics.positive_predictive_value_difference

---

::: skfair.metrics.positive_predictive_value_ratio

---

### Sufficiency — Negative Predictive Value

::: skfair.metrics.negative_predictive_value_difference

---

::: skfair.metrics.negative_predictive_value_ratio

---

### Sufficiency — False Discovery Rate

::: skfair.metrics.false_discovery_rate_difference

---

::: skfair.metrics.false_discovery_rate_ratio

---

### Sufficiency — False Omission Rate

::: skfair.metrics.false_omission_rate_difference

---

::: skfair.metrics.false_omission_rate_ratio

---

### Accuracy

::: skfair.metrics.accuracy_difference

---

::: skfair.metrics.accuracy_ratio

---

## Performance metrics

::: skfair.metrics.accuracy

---

::: skfair.metrics.balanced_accuracy

---

::: skfair.metrics.geometric_mean

---

::: skfair.metrics.precision

---

::: skfair.metrics.recall

---

::: skfair.metrics.f1_score

---

::: skfair.metrics.true_positive_rate

---

::: skfair.metrics.false_positive_rate

---

::: skfair.metrics.true_negative_rate

---

::: skfair.metrics.false_negative_rate

---

::: skfair.metrics.positive_predictive_value

---

::: skfair.metrics.negative_predictive_value

---

::: skfair.metrics.false_discovery_rate

---

::: skfair.metrics.false_omission_rate
