# scikit-fair

**Fairness-aware preprocessing for machine learning with a scikit-learn compatible API.**

scikit-fair (`skfair`) is a Python library that provides a suite of fairness-aware preprocessing algorithms for binary classification. It integrates seamlessly with scikit-learn pipelines and follows the imbalanced-learn API for sampling methods.

## What it does

Algorithmic fairness is concerned with preventing machine learning models from discriminating against individuals based on sensitive attributes such as race, sex, or age. scikit-fair tackles this at the **preprocessing stage**: the data is transformed before a classifier ever sees it, so any downstream sklearn estimator can benefit.

The library covers three families of techniques:

- **Weighting**: assign per-sample weights that a classifier uses via `sample_weight`
- **Resampling**: modify or synthesise samples to balance outcome distributions across groups
- **Feature transformation**: repair feature distributions to reduce disparate impact

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

Five group fairness metrics and six performance metrics are included. All follow a consistent signature: `metric(y_true, y_pred, sensitive_attr)`.

## Links

- [Installation](installation.md)
- [Quick Start](user_guide/quickstart.md)
- [API Reference](api/preprocessing.md)
