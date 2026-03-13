# scikit-fair

**Fairness-aware machine learning toolkit with a scikit-learn compatible API.**

scikit-fair (`skfair`) is a Python library for fairness-aware binary classification. It covers the full pipeline — preprocessing, evaluation, auditing, comparison, and experimentation — and integrates seamlessly with scikit-learn and imbalanced-learn workflows.

**Documentation**: [https://jmcfig.github.io/scikit-fair/](https://jmcfig.github.io/scikit-fair/)

---

## Installation

```bash
pip install scikit-fair
```

Or install from source:

```bash
git clone https://github.com/jmcfig/scikit-fair.git
cd scikit-fair
pip install -e .
```

**Requirements**: Python >= 3.9, numpy >= 1.22, pandas >= 1.5, scikit-learn >= 1.3, imbalanced-learn >= 0.12, cvxpy >= 1.3.

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

pipe = Pipeline([
    ("fair_smote", FairSmote(sens_attr="sex", random_state=42)),
    ("drop_sens", DropColumns("sex")), #optional
    ("classifier", LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

> **Tip:** We recommend always using `imblearn.pipeline.Pipeline` — it extends sklearn's Pipeline with `fit_resample` support, so it works with all scikit-fair methods (transformers, samplers, and meta-estimators) without needing to switch imports.

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

Nine group-fairness metrics and nine performance metrics share a unified signature: `metric(y_true, y_pred, sensitive_attr)`.

### Fairness metrics

| Function | Definition | Perfect value |
|---|---|---|
| `disparate_impact` | P(Y=1\|S=0) / P(Y=1\|S=1) | 1.0 |
| `statistical_parity_difference` | P(Y=1\|S=0) - P(Y=1\|S=1) | 0.0 |
| `equal_opportunity_difference` | TPR(S=0) - TPR(S=1) | 0.0 |
| `equal_opportunity_ratio` | TPR(S=0) / TPR(S=1) | 1.0 |
| `average_odds_difference` | 0.5 x [(FPR diff) + (TPR diff)] | 0.0 |
| `true_negative_rate_difference` | TNR(S=0) - TNR(S=1) | 0.0 |
| `false_negative_rate_difference` | FNR(S=0) - FNR(S=1) | 0.0 |
| `predictive_equality` | FPR(S=0) / FPR(S=1) | 1.0 |
| `accuracy_parity` | Acc(S=0) / Acc(S=1) | 1.0 |

### Performance metrics

`accuracy`, `true_positive_rate`, `false_positive_rate`, `true_negative_rate`, `false_negative_rate`, `balanced_accuracy`, `precision`, `recall`, `f1_score`.

```python
from skfair.metrics import (
    disparate_impact,
    statistical_parity_difference,
    equal_opportunity_difference,
    predictive_equality,
    accuracy,
    balanced_accuracy,
    precision,
    recall,
    f1_score,
)

sens = X_test["sex"].values
print(f"Accuracy:          {accuracy(y_test.values, y_pred):.3f}")
print(f"Balanced accuracy: {balanced_accuracy(y_test.values, y_pred):.3f}")
print(f"Precision:         {precision(y_test.values, y_pred):.3f}")
print(f"Recall:            {recall(y_test.values, y_pred):.3f}")
print(f"F1 score:          {f1_score(y_test.values, y_pred):.3f}")
print(f"Disparate impact:  {disparate_impact(y_test.values, y_pred, sens):.3f}")
print(f"Stat. parity diff: {statistical_parity_difference(y_test.values, y_pred, sens):.3f}")
print(f"Equal opp. diff:   {equal_opportunity_difference(y_test.values, y_pred, sens):.3f}")
print(f"Pred. equality:    {predictive_equality(y_test.values, y_pred, sens):.3f}")
```

---

## Datasets

Five standard fairness benchmarks are bundled.

| Loader | Samples | Features | Sensitive attribute | Label |
|---|---|---|---|---|
| `load_adult` | 48 842 | 14 | `sex` (1 = male) | income > 50k |
| `load_german` | 1 000 | 20 | `sex` | credit risk |
| `load_heart_disease` | 740 | 13 | `sex` | heart disease |
| `load_compas` | ~7 214 | 11 | `sex`, `race` | two-year recidivism |
| `load_ricci` | 118 | 5 | `Race` | promotion eligibility |

```python
from skfair.datasets import load_adult, load_german, load_heart_disease, load_compas, load_ricci

X, y = load_adult(preprocessed=True)
X, y = load_german()
X, y = load_heart_disease()
X, y = load_compas()
X, y = load_ricci()
```

---

## Audit

The `audit` module provides data-level and prediction-level fairness analysis.

### BiasAuditor — pre-model data analysis

Examines sensitive-group proportions, target rates, and feature distributions before training.

```python
from skfair.audit import BiasAuditor

auditor = BiasAuditor(X_train, y_train, sens_attr="sex")
print(auditor.group_proportions())
print(auditor.target_rate_by_group())
auditor.plot_summary()
```

### FairnessAuditor — post-model prediction analysis

Evaluates how fair a model's predictions are across groups.

```python
from skfair.audit import FairnessAuditor

fa = FairnessAuditor(y_test.values, y_pred, X_test["sex"].values)
print(fa.performance_by_group())
print(fa.fairness_metrics())
fa.plot_fairness_radar()
```

---

## Comparison

The `comparison` module provides a `ComparisonReport` for comparing multiple preprocessing methods across datasets and classifiers.

`ComparisonReport` expects a DataFrame with the following columns:

| Column | Required | Description |
|---|---|---|
| `dataset` | yes | Dataset name (e.g. `"adult"`, `"compas"`) |
| `method` | yes | Preprocessing method name (e.g. `"Massaging"`, `"FairSmote"`) |
| `classifier` | yes | Classifier name (e.g. `"LogisticRegression"`) |
| `{metric}` | yes (at least one) | Value for each metric (e.g. `accuracy`, `spd`) |
| `{metric}_std` | no | Standard deviation — included when `Experiment(std=True)`, not used by plots |

This is the format returned by `Experiment.run()`, but you can also build it manually.

```python
from skfair.comparison import ComparisonReport

report = ComparisonReport(results_df)

# Summary tables — pivot of metric means per method, averaged over classifiers
tables = report.summary_tables()

# Performance bar charts (accuracy, F1, etc.) grouped by method and classifier
report.plot_performance()

# Fairness bars averaged over classifiers for a single metric
report.plot_fairness_averaged(metric="spd")

# Fairness bars broken down per classifier
report.plot_fairness_detailed(metric="spd")

# Accuracy vs |fairness| scatter — ideally a method sits in the top-right corner
report.plot_tradeoff(fairness_metric="spd", performance_metric="accuracy")

# Heatmap ranking methods per dataset across all metrics
report.plot_ranking()

# Or generate all plots at once
report.plot_all(fairness_metric="spd")
```

---

## Experimentation

The `experimentation` module automates dataset x method x classifier experiments with cross-validation.

```python
from skfair.experimentation import Experiment

exp = Experiment(
    datasets=["adult", "compas"],
    methods=["Massaging", "FairSmote", "Reweighing"],
    n_splits=5,
)
results = exp.run()

# Generate a ComparisonReport
report = exp.to_report()
report.plot_performance()
```

Experiments can also be configured via XML files:

```python
exp = Experiment.from_xml("config.xml")
results = exp.run()
```

---

## Example notebooks

The [`examples/`](examples/) folder contains step-by-step Jupyter notebooks that walk through every module:

| Notebook | Description |
|---|---|
| [`01_datasets`](examples/01_datasets.ipynb) | Loading, exploring, and preprocessing the bundled datasets |
| [`02_methods`](examples/02_methods.ipynb) | Using fairness methods — transformers, samplers, and meta-estimators |
| [`03_audit`](examples/03_audit.ipynb) | Pre-model bias analysis and post-model fairness auditing |
| [`04_comparison`](examples/04_comparison.ipynb) | Comparing methods side-by-side with `ComparisonReport` |
| [`05_experiment`](examples/05_experiment.ipynb) | Running cross-validated experiments with `Experiment` |
| [`05a_experiment_config`](examples/05a_experiment_config.ipynb) | Configuring experiments from Python and XML |
| [`05b_custom_datasets`](examples/05b_custom_datasets.ipynb) | Using custom (user-provided) datasets in experiments |
| [`06_benchmark`](examples/06_benchmark.ipynb) | Full-scale benchmark driven by an XML config file |

---

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
