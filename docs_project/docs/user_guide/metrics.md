# Metrics

`skfair.metrics` provides group fairness metrics and standard performance metrics for binary classification.

All metrics share a consistent signature:

```python
metric(y_true, y_pred, sensitive_attr) -> float
```

The `sensitive_attr` array must be binary: **1 = privileged group**, **0 = unprivileged group**.

Fairness metrics are organised in **counterpart pairs**: every base measure exposes both a *difference* form (ideal = 0) and a *ratio / parity* form (ideal = 1). Pick the form that suits your reporting:

| Base measure | Difference (ideal 0) | Ratio / Parity (ideal 1) |
|---|---|---|
| Positive prediction rate | `statistical_parity_difference` | `disparate_impact` |
| TPR | `equal_opportunity_difference` | `equal_opportunity_ratio` |
| FPR | `false_positive_rate_difference` | `predictive_equality` (alias: `false_positive_rate_parity`) |
| TNR | `true_negative_rate_difference` | `true_negative_rate_parity` |
| FNR | `false_negative_rate_difference` | `false_negative_rate_parity` |
| Accuracy | `accuracy_difference` | `accuracy_parity` |
| FPR + TPR (combined) | `average_odds_difference` | `average_odds_ratio` |

---

## Fairness metrics

### Positive prediction rate

#### Statistical Parity Difference (SPD)

```
SPD = P(Y=1 | S=0) - P(Y=1 | S=1)
```

Difference in positive prediction rates.

- **Perfect fairness**: 0.0
- Negative values indicate the unprivileged group receives fewer positive predictions.

```python
from skfair.metrics import statistical_parity_difference

spd = statistical_parity_difference(y_true, y_pred, sensitive_attr)
```

#### Disparate Impact (DI)

```
DI = P(Y=1 | S=0) / P(Y=1 | S=1)
```

Ratio of positive prediction rates between the unprivileged and privileged groups.

- **Perfect fairness**: 1.0
- **80% rule threshold**: 0.8 (below this is considered discriminatory by some legal standards)

```python
from skfair.metrics import disparate_impact

di = disparate_impact(y_true, y_pred, sensitive_attr)
```

---

### True Positive Rate

#### Equal Opportunity Difference (EOD)

```
EOD = TPR_unpriv - TPR_priv
```

Difference in true positive rates (recall) between groups.

- **Perfect fairness**: 0.0

```python
from skfair.metrics import equal_opportunity_difference

eod = equal_opportunity_difference(y_true, y_pred, sensitive_attr)
```

#### Equal Opportunity Ratio (EOR)

```
EOR = TPR_unpriv / TPR_priv
```

Ratio of true positive rates between groups.

- **Perfect fairness**: 1.0

```python
from skfair.metrics import equal_opportunity_ratio

eor = equal_opportunity_ratio(y_true, y_pred, sensitive_attr)
```

---

### False Positive Rate

#### False Positive Rate Difference (FPRD)

```
FPRD = FPR_unpriv - FPR_priv
```

Difference in false positive rates between groups.

- **Perfect fairness**: 0.0

```python
from skfair.metrics import false_positive_rate_difference

fprd = false_positive_rate_difference(y_true, y_pred, sensitive_attr)
```

#### Predictive Equality (PE)

```
PE = FPR_unpriv / FPR_priv
```

Ratio of false positive rates between groups. Also exported as
`false_positive_rate_parity`.

- **Perfect fairness**: 1.0

```python
from skfair.metrics import predictive_equality

pe = predictive_equality(y_true, y_pred, sensitive_attr)
```

---

### True Negative Rate

#### True Negative Rate Difference (TNRD)

```
TNRD = TNR_unpriv - TNR_priv
```

Difference in true negative rates (specificity) between groups.

- **Perfect fairness**: 0.0

```python
from skfair.metrics import true_negative_rate_difference

tnrd = true_negative_rate_difference(y_true, y_pred, sensitive_attr)
```

#### True Negative Rate Parity (TNRP)

```
TNRP = TNR_unpriv / TNR_priv
```

Ratio of true negative rates between groups.

- **Perfect fairness**: 1.0

```python
from skfair.metrics import true_negative_rate_parity

tnrp = true_negative_rate_parity(y_true, y_pred, sensitive_attr)
```

---

### False Negative Rate

#### False Negative Rate Difference (FNRD)

```
FNRD = FNR_unpriv - FNR_priv
```

Difference in false negative rates between groups.

- **Perfect fairness**: 0.0

```python
from skfair.metrics import false_negative_rate_difference

fnrd = false_negative_rate_difference(y_true, y_pred, sensitive_attr)
```

#### False Negative Rate Parity (FNRP)

```
FNRP = FNR_unpriv / FNR_priv
```

Ratio of false negative rates between groups.

- **Perfect fairness**: 1.0

```python
from skfair.metrics import false_negative_rate_parity

fnrp = false_negative_rate_parity(y_true, y_pred, sensitive_attr)
```

---

### Accuracy

#### Accuracy Difference (AD)

```
AD = Acc_unpriv - Acc_priv
```

Difference in per-group accuracy.

- **Perfect fairness**: 0.0

```python
from skfair.metrics import accuracy_difference

ad = accuracy_difference(y_true, y_pred, sensitive_attr)
```

#### Accuracy Parity (AP)

```
AP = Acc_unpriv / Acc_priv
```

Ratio of accuracy between groups.

- **Perfect fairness**: 1.0

```python
from skfair.metrics import accuracy_parity

ap = accuracy_parity(y_true, y_pred, sensitive_attr)
```

---

### Combined (FPR + TPR)

#### Average Odds Difference (AOD)

```
AOD = 0.5 * [(FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv)]
```

Average of the FPR difference and TPR difference across groups. Captures both error rate equity and true positive equity.

- **Perfect fairness**: 0.0

```python
from skfair.metrics import average_odds_difference

aod = average_odds_difference(y_true, y_pred, sensitive_attr)
```

#### Average Odds Ratio (AOR)

```
AOR = 0.5 * [(FPR_unpriv / FPR_priv) + (TPR_unpriv / TPR_priv)]
```

Average of the FPR ratio and TPR ratio across groups. Returns NaN if either
component ratio is undefined (the privileged-group rate is zero while the
unprivileged-group rate is positive).

- **Perfect fairness**: 1.0

```python
from skfair.metrics import average_odds_ratio

aor = average_odds_ratio(y_true, y_pred, sensitive_attr)
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
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `f1_score` | 2 * precision * recall / (precision + recall) |

```python
from skfair.metrics import accuracy, balanced_accuracy, true_positive_rate, precision, recall, f1_score

print(accuracy(y_true, y_pred))
print(balanced_accuracy(y_true, y_pred))
print(true_positive_rate(y_true, y_pred))
print(precision(y_true, y_pred))
print(recall(y_true, y_pred))
print(f1_score(y_true, y_pred))
```

---

## Evaluating a preprocessing method

```python
from skfair.metrics import (
    accuracy,
    disparate_impact,
    statistical_parity_difference,
    equal_opportunity_difference,
    predictive_equality,
)

sens = X_test["sex"].values

def report(label, y_true, y_pred, sens):
    print(f"--- {label} ---")
    print(f"  Accuracy : {accuracy(y_true, y_pred):.3f}")
    print(f"  DI       : {disparate_impact(y_true, y_pred, sens):.3f}  (ideal 1.0)")
    print(f"  SPD      : {statistical_parity_difference(y_true, y_pred, sens):.3f}  (ideal 0.0)")
    print(f"  EOD      : {equal_opportunity_difference(y_true, y_pred, sens):.3f}  (ideal 0.0)")
    print(f"  PE       : {predictive_equality(y_true, y_pred, sens):.3f}  (ideal 1.0)")

report("Baseline", y_test.values, y_pred_base, sens)
report("After Massaging", y_test.values, y_pred_fair, sens)
```
