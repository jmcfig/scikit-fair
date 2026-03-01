# Quick Start

This guide walks through a minimal end-to-end example: load a dataset, apply a fairness preprocessor, train a classifier, and evaluate both performance and fairness.

## Load data

```python
from skfair.datasets import load_adult

X, y = load_adult(return_X_y=True, as_frame=True)
print(X.shape)       # (48842, 14)
print(X["sex"].value_counts())
```

The Adult census dataset contains a binary `sex` attribute (1 = male / privileged, 0 = female / unprivileged) and a binary income label.

## Baseline: no preprocessing

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skfair.metrics import disparate_impact, statistical_parity_difference, accuracy

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

sens = X_test["sex"].values
print(f"Accuracy:  {accuracy(y_test.values, y_pred):.3f}")
print(f"DI:        {disparate_impact(y_test.values, y_pred, sens):.3f}")  # ideally 1.0
print(f"SPD:       {statistical_parity_difference(y_test.values, y_pred, sens):.3f}")  # ideally 0.0
```

## Apply Massaging

`Massaging` is a label-modification technique that promotes unprivileged positive candidates and demotes privileged negative ones until the discrimination is minimised.

```python
from skfair.preprocessing import Massaging

sampler = Massaging(sens_attr="sex", priv_group=1)
X_fair, y_fair = sampler.fit_resample(X_train, y_train)

clf_fair = LogisticRegression(max_iter=1000)
clf_fair.fit(X_fair, y_fair)
y_pred_fair = clf_fair.predict(X_test)

print(f"Accuracy:  {accuracy(y_test.values, y_pred_fair):.3f}")
print(f"DI:        {disparate_impact(y_test.values, y_pred_fair, sens):.3f}")
print(f"SPD:       {statistical_parity_difference(y_test.values, y_pred_fair, sens):.3f}")
```

## Use Reweighing in a Pipeline

`Reweighing` does not change samples — it returns per-sample weights. Use the `ReweighingClassifier` wrapper to fit it inside any sklearn-compatible workflow:

```python
from skfair.preprocessing import ReweighingClassifier
from sklearn.linear_model import LogisticRegression

clf = ReweighingClassifier(
    estimator=LogisticRegression(max_iter=1000),
    sens_attr="sex",
    priv_group=1,
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

## Next steps

- [Preprocessing guide](preprocessing.md) — all algorithms explained
- [Metrics guide](metrics.md) — fairness and performance metrics
- [API Reference](../api/preprocessing.md)
