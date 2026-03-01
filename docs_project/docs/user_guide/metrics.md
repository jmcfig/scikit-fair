# Metrics

`skfair.metrics` provides group fairness metrics and standard performance metrics for binary classification.

All metrics share a consistent signature:

```python
metric(y_true, y_pred, sensitive_attr) -> float
```

The `sensitive_attr` array must be binary: **1 = privileged group**, **0 = unprivileged group**.

---

## Fairness metrics

### Disparate Impact (DI)

```
DI = P(Ŷ=1 | S=0) / P(Ŷ=1 | S=1)
```

Ratio of positive prediction rates between the unprivileged and privileged groups.

- **Perfect fairness**: 1.0
- **80% rule threshold**: 0.8 (below this is considered discriminatory by some legal standards)

```python
from skfair.metrics import disparate_impact

di = disparate_impact(y_true, y_pred, sensitive_attr)
```

### Statistical Parity Difference (SPD)

```
SPD = P(Ŷ=1 | S=0) - P(Ŷ=1 | S=1)
```

Difference in positive prediction rates.

- **Perfect fairness**: 0.0
- Negative values indicate the unprivileged group receives fewer positive predictions.

```python
from skfair.metrics import statistical_parity_difference

spd = statistical_parity_difference(y_true, y_pred, sensitive_attr)
```

### Equal Opportunity Difference (EOD)

```
EOD = TPR_unpriv - TPR_priv
```

Difference in true positive rates (recall) between groups.

- **Perfect fairness**: 0.0

```python
from skfair.metrics import equal_opportunity_difference

eod = equal_opportunity_difference(y_true, y_pred, sensitive_attr)
```

### Average Odds Difference (AOD)

```
AOD = 0.5 * [(FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv)]
```

Average of the FPR difference and TPR difference across groups. Captures both error rate equity and true positive equity.

- **Perfect fairness**: 0.0

```python
from skfair.metrics import average_odds_difference

aod = average_odds_difference(y_true, y_pred, sensitive_attr)
```

### True Negative Rate Difference (TNRD)

```
TNRD = TNR_unpriv - TNR_priv
```

Difference in true negative rates (specificity) between groups.

- **Perfect fairness**: 0.0

```python
from skfair.metrics import true_negative_rate_difference

tnrd = true_negative_rate_difference(y_true, y_pred, sensitive_attr)
```

---

## Performance metrics

These are group-agnostic wrappers that take `(y_true, y_pred)`.

| Function | Formula |
|---|---|
| `accuracy` | (TP + TN) / N |
| `true_positive_rate` | TP / (TP + FN) |
| `false_positive_rate` | FP / (FP + TN) |
| `true_negative_rate` | TN / (TN + FP) |
| `false_negative_rate` | FN / (FN + TP) |
| `balanced_accuracy` | 0.5 * (TPR + TNR) |

```python
from skfair.metrics import accuracy, balanced_accuracy, true_positive_rate

print(accuracy(y_true, y_pred))
print(balanced_accuracy(y_true, y_pred))
print(true_positive_rate(y_true, y_pred))
```

---

## Evaluating a preprocessing method

```python
from skfair.metrics import (
    accuracy,
    disparate_impact,
    statistical_parity_difference,
    equal_opportunity_difference,
)

sens = X_test["sex"].values

def report(label, y_true, y_pred, sens):
    print(f"--- {label} ---")
    print(f"  Accuracy : {accuracy(y_true, y_pred):.3f}")
    print(f"  DI       : {disparate_impact(y_true, y_pred, sens):.3f}  (ideal 1.0)")
    print(f"  SPD      : {statistical_parity_difference(y_true, y_pred, sens):.3f}  (ideal 0.0)")
    print(f"  EOD      : {equal_opportunity_difference(y_true, y_pred, sens):.3f}  (ideal 0.0)")

report("Baseline", y_test.values, y_pred_base, sens)
report("After Massaging", y_test.values, y_pred_fair, sens)
```
