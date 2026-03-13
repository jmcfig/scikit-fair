# scikit-fair

**Fairness-aware machine learning toolkit with a scikit-learn compatible API.**

scikit-fair (`skfair`) is a Python library for fairness-aware binary classification. It covers the full pipeline — preprocessing, evaluation, auditing, comparison, and experimentation — and integrates seamlessly with scikit-learn and imbalanced-learn workflows.

## What it does

Algorithmic fairness is concerned with preventing machine learning models from discriminating against individuals based on sensitive attributes such as race, sex, or age. scikit-fair provides tools for every stage of the fairness workflow:

- **Preprocessing**: transform data before training — weighting, resampling, and feature transformation techniques
- **Metrics**: nine group-fairness metrics and nine performance metrics with a unified API
- **Datasets**: five standard fairness benchmark datasets with convenient loaders
- **Audit**: pre-model data analysis (`BiasAuditor`) and post-model fairness evaluation (`FairnessAuditor`)
- **Comparison**: visual comparison of multiple preprocessing methods across datasets and classifiers (`ComparisonReport`)
- **Experimentation**: automated dataset × method × classifier experiments with cross-validation (`Experiment`)

## At a glance

```python
from skfair.preprocessing import Massaging
from skfair.metrics import disparate_impact, statistical_parity_difference
from skfair.datasets import load_adult

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_adult(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reduce discrimination via label massaging
sampler = Massaging(sens_attr="sex", priv_group=1)
X_fair, y_fair = sampler.fit_resample(X_train, y_train)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_fair, y_fair)
y_pred = clf.predict(X_test)

# Measure improvement
print(disparate_impact(y_test.values, y_pred, X_test["sex"].values))
```

## Algorithms

| Class | Type | Reference |
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

## Metrics

Nine group fairness metrics and nine performance metrics are included. All follow a consistent signature: `metric(y_true, y_pred, sensitive_attr)`.

## Example notebooks

The [`examples/`](https://github.com/jmcfig/scikit-fair/tree/main/examples) folder contains step-by-step Jupyter notebooks that walk through every module:

| Notebook | Description |
|---|---|
| `01_datasets` | Loading, exploring, and preprocessing the bundled datasets |
| `02_methods` | Using fairness methods — transformers, samplers, and meta-estimators |
| `03_audit` | Pre-model bias analysis and post-model fairness auditing |
| `04_comparison` | Comparing methods side-by-side with `ComparisonReport` |
| `05_experiment` | Running cross-validated experiments with `Experiment` |
| `05a_experiment_config` | Configuring experiments from Python and YAML |
| `05b_custom_datasets` | Using custom (user-provided) datasets in experiments |
| `06_benchmark` | Full-scale benchmark driven by a YAML config file |

## Links

- [Installation](installation.md)
- [Quick Start](user_guide/quickstart.md)
- [Preprocessing](user_guide/preprocessing.md)
- [Metrics](user_guide/metrics.md)
- [Datasets](user_guide/datasets.md)
- [Audit](user_guide/audit.md)
- [Comparison](user_guide/comparison.md)
- [Experimentation](user_guide/experimentation.md)
- [API Reference](api/preprocessing.md)
