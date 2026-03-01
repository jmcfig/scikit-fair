# scikit-fair

**Fairness-aware preprocessing for machine learning with a scikit-learn compatible API.**

scikit-fair (`skfair`) is a Python library providing a suite of fairness preprocessing algorithms for binary classification. It integrates seamlessly with scikit-learn pipelines and follows the imbalanced-learn API for sampling methods, so it slots into any existing sklearn workflow without friction.

This is a current implementation which will be expanded in the near future.

---

## Installation from source

```bash
git clone https://github.com/jmcfig/scikit-fair.git
cd scikit-fair
pip install -e .
```

**Requirements**: Python ≥ 3.9, numpy ≥ 1.22, pandas ≥ 1.5, scikit-learn ≥ 1.3, imbalanced-learn ≥ 0.12, cvxpy ≥ 1.3.

---

## Quick start

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from skfair.datasets import load_adult
from skfair.preprocessing import Massaging
from skfair.metrics import accuracy, disparate_impact, statistical_parity_difference

# 1. Load data
X, y = load_adult(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Baseline — no fairness preprocessing
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

sens = X_test["sex"].values
print(f"Baseline — Accuracy: {accuracy(y_test.values, y_pred):.3f}  "
      f"DI: {disparate_impact(y_test.values, y_pred, sens):.3f}  "
      f"SPD: {statistical_parity_difference(y_test.values, y_pred, sens):.3f}")

# 3. Apply Massaging to reduce label bias
sampler = Massaging(sens_attr="sex", priv_group=1)
X_fair, y_fair = sampler.fit_resample(X_train, y_train)

clf_fair = LogisticRegression(max_iter=1000)
clf_fair.fit(X_fair, y_fair)
y_pred_fair = clf_fair.predict(X_test)

print(f"Fair    — Accuracy: {accuracy(y_test.values, y_pred_fair):.3f}  "
      f"DI: {disparate_impact(y_test.values, y_pred_fair, sens):.3f}  "
      f"SPD: {statistical_parity_difference(y_test.values, y_pred_fair, sens):.3f}")
```

---

## Algorithms

| Class | Family | Reference |
|---|---|---|
| `Reweighing` | Weighting | Kamiran & Calders (2012) |
| `FairBalance` | Weighting | Yu et al. (2024) |
| `ReweighingClassifier` | Meta-estimator | — |
| `FairBalanceClassifier` | Meta-estimator | — |
| `Massaging` | Label modification | Kamiran & Calders (2012) |
| `FairwayRemover` | Label modification | Fairway (2019) |
| `FairOversampling` | Oversampling | Dablan et al. |
| `FairSmote` | Oversampling | Chakraborty et al. (2021) |
| `FAWOS` | Oversampling | Salazar et al. (2021) |
| `HeterogeneousFOS` | Oversampling | Sonoda et al. (2023) |
| `GeometricFairnessRepair` | Feature transformation | Feldman et al. (2015) |
| `OptimizedPreprocessing` | Feature transformation | Calmon et al. (2017) |
| `LearningFairRepresentations` | Feature transformation | Zemel et al. (2013) |
| `FairMask` | Meta-estimator | Peng et al. (2021) |
| `IntersectionalBinarizer` | Utility | — |
| `DropColumns` | Utility | — |

---

## Usage patterns

Each family of algorithms has its own API contract.

### Samplers — `fit_resample(X, y)`

Label-modification and oversampling methods return a resampled dataset. They extend `imblearn.BaseSampler` and work directly inside an `imblearn.Pipeline`.

```python
from skfair.preprocessing import FairSmote

sampler = FairSmote(sens_attr="sex", random_state=0)
X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
```

### Weighting methods — `fit_transform(X, y)`

`Reweighing` and `FairBalance` return the original `X` unchanged alongside a weight Series. Pass the weights to your classifier via `sample_weight`.

```python
from skfair.preprocessing import Reweighing

rw = Reweighing(sens_attr="sex", priv_group=1)
X_unchanged, weights = rw.fit_transform(X_train, y_train)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_unchanged, y_train, sample_weight=weights)
```

### Classifier wrappers — standard `fit` / `predict`

`ReweighingClassifier` and `FairBalanceClassifier` encapsulate the weighting step inside a full sklearn-compatible classifier, including `sample_weight` handling.

```python
from skfair.preprocessing import ReweighingClassifier

clf = ReweighingClassifier(
    estimator=LogisticRegression(max_iter=1000),
    sens_attr="sex",
    priv_group=1,
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

### Feature transformers — `fit_transform(X)`

`GeometricFairnessRepair`, `OptimizedPreprocessing`, and `LearningFairRepresentations` transform `X` directly and slot into `sklearn.Pipeline` as standard transformers.

```python
from skfair.preprocessing import GeometricFairnessRepair

repair = GeometricFairnessRepair(
    sensitive_attribute="sex",
    repair_columns=["age", "hours-per-week"],
    lambda_param=1.0,
)
X_repaired = repair.fit_transform(X_train)
```

### example Pipeline

Combine preprocessing with downstream estimators, optionally using `DropColumns` to remove the sensitive attribute just before the classifier.

```python
from imblearn.pipeline import Pipeline
from skfair.preprocessing import FairSmote, DropColumns

pipe = ImbPipeline([
    ("fair_smote", FairSmote(sens_attr="sex", random_state=42)),
    ("drop_sens", DropColumns("sex")), #optional
    ("classifier", LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### Intersectional privilege

Define complex, multi-column privilege criteria with `IntersectionalBinarizer`.

```python
from skfair.preprocessing import IntersectionalBinarizer

binarizer = IntersectionalBinarizer(
    privileged_definition={"race": "White", "sex": "Male"},
    group_col_name="_is_privileged",
)
X_with_group = binarizer.fit_transform(X_train)
```

---

## Metrics

Five group-fairness metrics (and six performance metrics) share a unified signature: `metric(y_true, y_pred, sensitive_attr)`.

### Fairness metrics

| Function | Definition | Perfect value |
|---|---|---|
| `disparate_impact` | P(Ŷ=1\|S=0) / P(Ŷ=1\|S=1) | 1.0 |
| `statistical_parity_difference` | P(Ŷ=1\|S=0) − P(Ŷ=1\|S=1) | 0.0 |
| `equal_opportunity_difference` | TPR(S=0) − TPR(S=1) | 0.0 |
| `average_odds_difference` | 0.5 × [(FPR diff) + (TPR diff)] | 0.0 |
| `true_negative_rate_difference` | TNR(S=0) − TNR(S=1) | 0.0 |

### Performance metrics

`accuracy`, `true_positive_rate`, `false_positive_rate`, `true_negative_rate`, `false_negative_rate`, `balanced_accuracy`.

```python
from skfair.metrics import (
    disparate_impact,
    statistical_parity_difference,
    equal_opportunity_difference,
    accuracy,
    balanced_accuracy,
)

sens = X_test["sex"].values
print(f"Accuracy:          {accuracy(y_test.values, y_pred):.3f}")
print(f"Balanced accuracy: {balanced_accuracy(y_test.values, y_pred):.3f}")
print(f"Disparate impact:  {disparate_impact(y_test.values, y_pred, sens):.3f}")
print(f"Stat. parity diff: {statistical_parity_difference(y_test.values, y_pred, sens):.3f}")
print(f"Equal opp. diff:   {equal_opportunity_difference(y_test.values, y_pred, sens):.3f}")
```

---

## Datasets

Three standard fairness benchmarks are bundled so far. 

| Loader | Samples | Features | Sensitive attribute | Label |
|---|---|---|---|---|
| `load_adult` | 48 842 | 14 | `sex` (1 = male) | income > 50k |
| `load_german` | 1 000 | 20 | `sex` | credit risk |
| `load_heart_disease` | 740 | 13 | `sex` | heart disease |

```python
from skfair.datasets import load_adult, load_german, load_heart_disease

X, y = load_adult(preprocessed=True) #load adult pipeline ready, with simple preprocessing
X, y = load_german()
X, y = load_heart_disease()
```

