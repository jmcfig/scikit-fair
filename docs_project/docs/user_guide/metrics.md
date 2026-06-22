# Metrics

`skfair.metrics` provides group fairness metrics and standard performance metrics for binary classification.

All metrics share a consistent signature:

```python
metric(y_true, y_pred, sensitive_attr) -> float
```

The `sensitive_attr` array must be binary: **1 = privileged group**, **0 = unprivileged group**.

Fairness metrics are organised in **counterpart pairs**: every base measure exposes both a *difference* form (suffix `_difference`, ideal = 0) and a *ratio* form (suffix `_ratio`, ideal = 1). Canonical names are the full descriptive forms; short aliases are also exported. Pick the form that suits your reporting:

| Family | Base measure | Difference (ideal 0) | Ratio (ideal 1) |
|---|---|---|---|
| Independence | Positive prediction rate | `statistical_parity_difference` (`spd`) | `disparate_impact` (`di`) |
| Separation | TPR | `true_positive_rate_difference` (`eod`) | `true_positive_rate_ratio` (`eor`) |
| Separation | FPR | `false_positive_rate_difference` (`fpr_diff`) | `false_positive_rate_ratio` (`fpr_ratio`) |
| Separation | TNR | `true_negative_rate_difference` (`tnr_diff`) | `true_negative_rate_ratio` (`tnr_ratio`) |
| Separation | FNR | `false_negative_rate_difference` (`fnr_diff`) | `false_negative_rate_ratio` (`fnr_ratio`) |
| Separation | FPR + TPR (combined odds) | `average_odds_difference` (`aod`) | `average_odds_ratio` (`aor`) |
| Sufficiency | PPV (predictive parity) | `positive_predictive_value_difference` (`ppv_diff`) | `positive_predictive_value_ratio` (`ppv_ratio`) |
| Sufficiency | NPV | `negative_predictive_value_difference` (`npv_diff`) | `negative_predictive_value_ratio` (`npv_ratio`) |
| Sufficiency | FDR | `false_discovery_rate_difference` (`fdr_diff`) | `false_discovery_rate_ratio` (`fdr_ratio`) |
| Sufficiency | FOR | `false_omission_rate_difference` (`for_diff`) | `false_omission_rate_ratio` (`for_ratio`) |
| Accuracy | Accuracy | `accuracy_difference` (`acc_diff`) | `accuracy_ratio` (`acc_ratio`) |

> **Naming policy:** canonical names always use a `_difference` or `_ratio` suffix — never `_parity`. The legacy `equal_opportunity_difference` / `equal_opportunity_ratio` names remain as aliases of the TPR pair.

---

## Redundancies in the grid

The package exposes each base measure in both a difference and a ratio form, and
includes some entries that are mathematically redundant. We keep them for
completeness and symmetry — so the grid reads uniformly and you can pick whichever
*lens* matches the harm you care about — not because every number is independent,
and not as an attempt to cover every fairness metric in the literature.

**Which differences are redundant.** Four base measures are exact complements of
another: `TNR = 1 − FPR`, `FNR = 1 − TPR`, `FDR = 1 − PPV`, `FOR = 1 − NPV`. The
difference of a complement is just the negated difference of its partner:

```
false_negative_rate_difference   = − true_positive_rate_difference        (FNR_diff = −TPR_diff)
true_negative_rate_difference     = − false_positive_rate_difference       (TNR_diff = −FPR_diff)
false_discovery_rate_difference   = − positive_predictive_value_difference (FDR_diff = −PPV_diff)
false_omission_rate_difference    = − negative_predictive_value_difference (FOR_diff = −NPV_diff)
```

So within each complementary pair you only ever need **one** difference; the other
carries no new information (it is literally computed as the sign-flip).

**Why keep them anyway — choosing the FNR lens vs the TPR lens.** The redundant
difference is kept because its *framing* matters. `TPR_diff` (equal opportunity)
asks "do qualified people get approved at equal rates?"; `FNR_diff` asks "are
qualified people *missed* at equal rates?" — same magnitude, opposite sign, but in
a screening or diagnostic setting where the harm is a **missed positive**, reading
the FNR gap is the more natural way to talk about it. Likewise FPR vs TNR, and
PPV vs FDR. Report the one whose direction-of-harm reads cleanly for your problem.

**Ratios are *not* redundant.** A ratio of `1 − x` values is not a function of the
ratio of the `x` values, so `true_negative_rate_ratio`, `false_negative_rate_ratio`,
`false_discovery_rate_ratio` and `false_omission_rate_ratio` each carry genuinely
new information that their partner ratio does not. The differences collapse; the
ratios do not.

**Aggregates.** `average_odds_difference` / `average_odds_ratio` are convenience
aggregates of the FPR and TPR components (equalized-odds reporting in one number).
They are derivable from their parts and are provided for compact reporting, not as
independent measurements.

---

## Quick benchmark

To evaluate several fairness metrics at once, use `benchmark`. With no `metrics`
argument it returns a sensible one-per-family subset; pass `metrics="all"` for
the full grid, or a list of names/aliases for a custom selection.

```python
from skfair.metrics import benchmark

benchmark(y_true, y_pred, sensitive_attr)
# one representative difference per family:
# {'statistical_parity_difference': ...,        # independence
#  'equal_opportunity_difference': ...,         # separation (TPR)
#  'false_positive_rate_difference': ...,       # separation (FPR)
#  'positive_predictive_value_difference': ...} # sufficiency

benchmark(y_true, y_pred, sensitive_attr, metrics=["spd", "eod", "ppv_ratio"])
benchmark(y_true, y_pred, sensitive_attr, metrics="all")
```

All metric metadata lives in a single registry, `skfair.metrics.REGISTRY`
(canonical name → `MetricSpec` with `func`, `kind`, `family`, `ideal`, `display`,
`aliases`). `METRICS` is the fairness-only view, and `DEFAULT_BENCHMARK_METRICS`
is the default subset. Everything else in the package (the auditor, comparison
reports, experiments) derives its metric information from this one registry.

---

## Fairness metrics

### Independence

#### Statistical Parity Difference (SPD)

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

#### Disparate Impact (DI)

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

---

### Separation — True Positive Rate

#### True Positive Rate Difference (alias `eod`, `equal_opportunity_difference`)

```
TPRD = TPR_unpriv - TPR_priv
```

Difference in true positive rates (recall) between groups.

- **Perfect fairness**: 0.0

```python
from skfair.metrics import true_positive_rate_difference

tprd = true_positive_rate_difference(y_true, y_pred, sensitive_attr)
```

#### True Positive Rate Ratio (alias `eor`, `equal_opportunity_ratio`)

```
TPRR = TPR_unpriv / TPR_priv
```

Ratio of true positive rates between groups.

- **Perfect fairness**: 1.0

```python
from skfair.metrics import true_positive_rate_ratio

tprr = true_positive_rate_ratio(y_true, y_pred, sensitive_attr)
```

---

### Separation — False Positive Rate

```
FPRD = FPR_unpriv - FPR_priv        FPRR = FPR_unpriv / FPR_priv
```

Difference / ratio of false positive rates between groups (FPR equity, sometimes
called *predictive equality*).

- **Perfect fairness**: 0.0 (difference), 1.0 (ratio)

```python
from skfair.metrics import false_positive_rate_difference, false_positive_rate_ratio

fprd = false_positive_rate_difference(y_true, y_pred, sensitive_attr)
fprr = false_positive_rate_ratio(y_true, y_pred, sensitive_attr)
```

---

### Separation — True Negative Rate

```
TNRD = TNR_unpriv - TNR_priv        TNRR = TNR_unpriv / TNR_priv
```

Difference / ratio of true negative rates (specificity) between groups.

- **Perfect fairness**: 0.0 (difference), 1.0 (ratio)

```python
from skfair.metrics import true_negative_rate_difference, true_negative_rate_ratio

tnrd = true_negative_rate_difference(y_true, y_pred, sensitive_attr)
tnrr = true_negative_rate_ratio(y_true, y_pred, sensitive_attr)
```

---

### Separation — False Negative Rate

```
FNRD = FNR_unpriv - FNR_priv        FNRR = FNR_unpriv / FNR_priv
```

Difference / ratio of false negative rates between groups.

- **Perfect fairness**: 0.0 (difference), 1.0 (ratio)

```python
from skfair.metrics import false_negative_rate_difference, false_negative_rate_ratio

fnrd = false_negative_rate_difference(y_true, y_pred, sensitive_attr)
fnrr = false_negative_rate_ratio(y_true, y_pred, sensitive_attr)
```

---

### Separation — Combined odds (FPR + TPR)

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

### Sufficiency

Sufficiency metrics condition on the **prediction** Ŷ rather than the true label
Y — they read the same confusion-matrix cells along the prediction axis.

| Base measure | Difference | Ratio |
|---|---|---|
| PPV (predictive parity) | `positive_predictive_value_difference` | `positive_predictive_value_ratio` |
| NPV | `negative_predictive_value_difference` | `negative_predictive_value_ratio` |
| FDR | `false_discovery_rate_difference` | `false_discovery_rate_ratio` |
| FOR | `false_omission_rate_difference` | `false_omission_rate_ratio` |

```
PPVD = PPV_unpriv - PPV_priv        PPVR = PPV_unpriv / PPV_priv
```

- **Perfect fairness**: 0.0 (difference), 1.0 (ratio)
- Difference/ratio metrics return NaN when a group has no predicted positives
  (PPV/FDR) or no predicted negatives (NPV/FOR), since the base measure is then
  undefined.

```python
from skfair.metrics import (
    positive_predictive_value_ratio,   # alias: ppv_ratio
    negative_predictive_value_ratio,   # alias: npv_ratio
    false_discovery_rate_ratio,        # alias: fdr_ratio
    false_omission_rate_ratio,         # alias: for_ratio
)

ppvr = positive_predictive_value_ratio(y_true, y_pred, sensitive_attr)
```

---

### Accuracy

```
AD = Acc_unpriv - Acc_priv        AR = Acc_unpriv / Acc_priv
```

Difference / ratio of per-group accuracy.

- **Perfect fairness**: 0.0 (difference), 1.0 (ratio)

```python
from skfair.metrics import accuracy_difference, accuracy_ratio

ad = accuracy_difference(y_true, y_pred, sensitive_attr)
ar = accuracy_ratio(y_true, y_pred, sensitive_attr)
```

---

## Performance metrics

These are group-agnostic wrappers that take `(y_true, y_pred)`.

| Function | Formula |
|---|---|
| `accuracy` | (TP + TN) / N |
| `balanced_accuracy` | 0.5 * (TPR + TNR) |
| `geometric_mean` | sqrt(TPR * TNR) |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `f1_score` | 2 * precision * recall / (precision + recall) |
| `true_positive_rate` | TP / (TP + FN) |
| `false_positive_rate` | FP / (FP + TN) |
| `true_negative_rate` | TN / (TN + FP) |
| `false_negative_rate` | FN / (FN + TP) |
| `positive_predictive_value` | TP / (TP + FP) |
| `negative_predictive_value` | TN / (TN + FN) |
| `false_discovery_rate` | FP / (FP + TP) |
| `false_omission_rate` | FN / (FN + TN) |

```python
from skfair.metrics import accuracy, balanced_accuracy, geometric_mean, true_positive_rate, precision, recall, f1_score

print(accuracy(y_true, y_pred))
print(balanced_accuracy(y_true, y_pred))
print(geometric_mean(y_true, y_pred))
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
    true_positive_rate_difference,
    false_positive_rate_ratio,
)

sens = X_test["sex"].values

def report(label, y_true, y_pred, sens):
    print(f"--- {label} ---")
    print(f"  Accuracy : {accuracy(y_true, y_pred):.3f}")
    print(f"  DI       : {disparate_impact(y_true, y_pred, sens):.3f}  (ideal 1.0)")
    print(f"  SPD      : {statistical_parity_difference(y_true, y_pred, sens):.3f}  (ideal 0.0)")
    print(f"  TPRD     : {true_positive_rate_difference(y_true, y_pred, sens):.3f}  (ideal 0.0)")
    print(f"  FPRR     : {false_positive_rate_ratio(y_true, y_pred, sens):.3f}  (ideal 1.0)")

report("Baseline", y_test.values, y_pred_base, sens)
report("After Massaging", y_test.values, y_pred_fair, sens)
```
