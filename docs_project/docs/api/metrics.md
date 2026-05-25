# skfair.metrics

```python
from skfair.metrics import <function_name>
```

Fairness metrics are exposed as **counterpart pairs**: each base measure
has both a *difference* form (ideal = 0) and a *ratio / parity* form
(ideal = 1).

---

## Fairness metrics

### Positive prediction rate

::: skfair.metrics.statistical_parity_difference

---

::: skfair.metrics.disparate_impact

---

### True Positive Rate

::: skfair.metrics.equal_opportunity_difference

---

::: skfair.metrics.equal_opportunity_ratio

---

### False Positive Rate

::: skfair.metrics.false_positive_rate_difference

---

::: skfair.metrics.predictive_equality

`false_positive_rate_parity` is exported as an alias of
`predictive_equality`.

---

### True Negative Rate

::: skfair.metrics.true_negative_rate_difference

---

::: skfair.metrics.true_negative_rate_parity

---

### False Negative Rate

::: skfair.metrics.false_negative_rate_difference

---

::: skfair.metrics.false_negative_rate_parity

---

### Accuracy

::: skfair.metrics.accuracy_difference

---

::: skfair.metrics.accuracy_parity

---

### Combined (FPR + TPR)

::: skfair.metrics.average_odds_difference

---

::: skfair.metrics.average_odds_ratio

---

## Performance metrics

::: skfair.metrics.accuracy

---

::: skfair.metrics.true_positive_rate

---

::: skfair.metrics.false_positive_rate

---

::: skfair.metrics.true_negative_rate

---

::: skfair.metrics.false_negative_rate

---

::: skfair.metrics.balanced_accuracy

---

::: skfair.metrics.precision

---

::: skfair.metrics.recall

---

::: skfair.metrics.f1_score
