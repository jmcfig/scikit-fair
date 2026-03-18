# Audit

The `skfair.audit` module provides two auditors for fairness analysis:

- **BiasAuditor** — pre-model, data-level disparity analysis
- **FairnessAuditor** — post-model, prediction-level fairness evaluation

---

## BiasAuditor

`BiasAuditor` examines the training data *before* any model is trained. It helps identify existing disparities in group representation and label distribution.

### Setup

```python
from skfair.datasets import load_adult
from skfair.audit import BiasAuditor
from sklearn.model_selection import train_test_split

X, y = load_adult(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

auditor = BiasAuditor(X_train, y_train, sens_attr="sex")
```

### Group proportions

How many samples belong to each group?

```python
auditor.group_proportions()
```

Returns a DataFrame with `count` and `proportion` columns, indexed by group value.

### Target rate by group

What fraction of each group has the positive label?

```python
auditor.target_rate_by_group()
```

Returns a DataFrame with `count` and `positive_rate` columns.

### Visualisations

```python
# Individual plots
auditor.plot_group_proportions()
auditor.plot_target_rates()
auditor.plot_feature_distribution("age")

# All-in-one summary (group proportions, target rates, and feature distributions)
auditor.plot_summary()
```

`plot_summary()` returns a list of `(fig, ax)` tuples covering group proportions, target rates, and distributions for all numeric features.

---

## FairnessAuditor

`FairnessAuditor` evaluates a model's predictions against the ground truth, broken down by sensitive group.

### Setup

```python
from sklearn.linear_model import LogisticRegression
from skfair.audit import FairnessAuditor

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

fa = FairnessAuditor(y_test.values, y_pred, X_test["sex"].values)
```

### Performance by group

```python
fa.performance_by_group()
```

Returns a DataFrame with rows for each metric (Accuracy, TPR, FPR, TNR, FNR) and columns for `unprivileged` and `privileged`.

### Fairness metrics

```python
fa.fairness_metrics()
```

Returns a DataFrame with all nine fairness metrics: Disparate Impact, Statistical Parity Diff, Equal Opportunity Diff, Equal Opportunity Ratio, Average Odds Diff, TNR Difference, FNR Difference, Predictive Equality, and Accuracy Parity.

### Visualisations

```python
# Grouped bar chart of performance metrics by group
fa.plot_performance_by_group()

# Horizontal bar chart of fairness metrics, colour-coded
fa.plot_fairness_metrics()

# Radar/spider chart of fairness metrics
fa.plot_fairness_radar()

# All three plots at once
fa.plot_summary()
```

---

## Combining BiasAuditor and FairnessAuditor

A typical workflow audits the data first, trains a model, then audits the predictions:

```python
from skfair.audit import BiasAuditor, FairnessAuditor
from skfair.datasets import load_adult
from skfair.preprocessing import Massaging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_adult(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Audit the data
bias = BiasAuditor(X_train, y_train, sens_attr="sex")
bias.plot_summary()

# 2. Train with fairness preprocessing
sampler = Massaging(sens_attr="sex", priv_group=1)
X_fair, y_fair = sampler.fit_resample(X_train, y_train)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_fair, y_fair)
y_pred = clf.predict(X_test)

# 3. Audit the predictions
fa = FairnessAuditor(y_test.values, y_pred, X_test["sex"].values)
print(fa.fairness_metrics())
fa.plot_fairness_radar()
```
