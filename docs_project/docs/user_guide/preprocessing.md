# Preprocessing

scikit-fair implements four families of fairness preprocessing algorithms.

---

## Weighting methods

These methods compute per-sample weights without modifying the feature matrix or labels. The weights are passed to downstream estimators via `sample_weight`.

### Reweighing

**Reference:** Kamiran & Calders (2012)

Assigns weights so that the weighted dataset exhibits statistical independence between the sensitive attribute `A` and the label `Y`:

```
weight(a, y) = P(A=a) * P(Y=y) / P(A=a, Y=y)
```

```python
from skfair.preprocessing import Reweighing

rw = Reweighing(sens_attr="sex", priv_group=1)
X_out, weights = rw.fit_transform(X, y)

clf.fit(X_out, y, sample_weight=weights)
```

### FairBalance

**Reference:** Yu, Chakraborty & Menzies (2024)

Balances class distribution within each demographic group:

```
weight(a, y) = |A=a| / |A=a, Y=y|
```

```python
from skfair.preprocessing import FairBalance

fb = FairBalance(sens_attr="sex", priv_group=1)
X_out, weights = fb.fit_transform(X, y)
```

A `variant` mode is available that additionally normalises by the overall group ratio.

### Wrapper classifiers

`ReweighingClassifier` and `FairBalanceClassifier` bundle the weighting step with any sklearn estimator:

```python
from skfair.preprocessing import ReweighingClassifier
from sklearn.ensemble import RandomForestClassifier

clf = ReweighingClassifier(
    estimator=RandomForestClassifier(),
    sens_attr="sex",
    priv_group=1,
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

## Label modification methods

These methods change class labels (not features) to reduce discrimination. The sample count stays the same (clean-sampling).

### Massaging

**Reference:** Kamiran & Calders (2012)

Uses logistic regression to rank candidates and swaps labels:

- **Promotion**: unprivileged samples with label 0, ranked by predicted probability of label 1
- **Demotion**: privileged samples with label 1, ranked by predicted probability of label 0

```python
from skfair.preprocessing import Massaging

sampler = Massaging(sens_attr="sex", priv_group=1)
X_fair, y_fair = sampler.fit_resample(X, y)
```

### FairwayRemover

Removes "ambiguous" samples where group-specific models trained on privileged and unprivileged subsets disagree. Only samples both models agree on are retained.

```python
from skfair.preprocessing import FairwayRemover

remover = FairwayRemover(sens_attr="sex", priv_group=1)
X_clean, y_clean = remover.fit_resample(X, y)
```

---

## Oversampling methods

These methods generate synthetic samples to bring all (class × group) subgroup sizes into balance.

### FairOversampling

Per-group class balancing using SMOTE interpolation. Within each protected group, the minority class is oversampled independently.

```python
from skfair.preprocessing import FairOversampling

fos = FairOversampling(sens_attr="sex", priv_group=1)
X_res, y_res = fos.fit_resample(X, y)
```

### FairSmote

Balances all four (class × group) subgroups using Differential Evolution-style mutation:

```
x_new = x_parent + F * (x_neighbor1 - x_neighbor2)
```

```python
from skfair.preprocessing import FairSmote

fs = FairSmote(sens_attr="sex", priv_group=1, cr=0.8, f=0.8)
X_res, y_res = fs.fit_resample(X, y)
```

### FAWOS

**Reference:** Salazar et al. (2021)

Typology-based weighted oversampling. Samples are classified by their KNN neighbourhood:

| Type | Same-type neighbours | Sampling weight |
|---|---|---|
| Safe | 4–5 | High |
| Borderline | 2–3 | Medium |
| Rare | 1 | Low |
| Outlier | 0 | Excluded |

```python
from skfair.preprocessing import FAWOS

fawos = FAWOS(sens_attr="sex", priv_group=1)
X_res, y_res = fawos.fit_resample(X, y)
```

### HeterogeneousFOS

**Reference:** Sonoda et al. (2023)

Uses heterogeneous clusters for interpolation:

- **H_y**: class-heterogeneous neighbours (different class, same group)
- **H_g**: group-heterogeneous neighbours (same class, different group)

Bernoulli probability determines which cluster to use for each synthetic sample.

```python
from skfair.preprocessing import HeterogeneousFOS

hfos = HeterogeneousFOS(sens_attr="sex", priv_group=1)
X_res, y_res = hfos.fit_resample(X, y)
```

---

## Feature transformation methods

These methods modify the feature matrix itself (they are sklearn `TransformerMixin`s).

### GeometricFairnessRepair

**Reference:** Feldman et al. (2015)

Repairs feature distributions using quantile buckets. Each non-sensitive feature is mapped towards a shared "median" distribution:

```
x_repaired = (1 - λ) * x_original + λ * x_repaired_value
```

`lambda_param=0.0` leaves the data unchanged; `lambda_param=1.0` applies full repair.

```python
from skfair.preprocessing import GeometricFairnessRepair

repair = GeometricFairnessRepair(sensitive_attribute="sex", lambda_param=0.8)
X_repaired = repair.fit_transform(X)
```

### OptimizedPreprocessing

**Reference:** Calmon et al. (2017)

Solves a convex optimisation problem to find a joint transformation of features and labels that minimises discrimination while preserving data utility.

```python
from skfair.preprocessing import OptimizedPreprocessing

op = OptimizedPreprocessing(
    sensitive_attribute="sex",
    epsilon=0.05,
)
X_out, y_out = op.fit_transform(X, y)
```

!!! warning
    Small datasets with tight `epsilon` can make the optimisation infeasible. Use `epsilon >= 0.05` and ensure you have enough samples in each subgroup.

!!! note
    `OptimizedPreprocessing` requires all features to be discrete (categorical). Because of this specific data requirement, it may be excluded from automated benchmarks, which typically use datasets with mixed continuous/discrete features.

### LearningFairRepresentations

**Reference:** Zemel et al. (2013)

Learns a fair intermediate representation by optimising three objectives simultaneously: prediction accuracy, statistical parity, and reconstruction fidelity.

```python
from skfair.preprocessing import LearningFairRepresentations

lfr = LearningFairRepresentations(sensitive_attribute="sex", k=10)
Z = lfr.fit_transform(X, y)
```

### FairMask

**Reference:** Peng et al. (2021)

A meta-estimator that masks sensitive attribute values at inference time. During training it builds extrapolation models; at prediction time the sensitive values are replaced with synthetic estimates:

```python
from skfair.preprocessing import FairMask
from sklearn.linear_model import LogisticRegression

clf = FairMask(
    estimator=LogisticRegression(max_iter=1000),
    sensitive_attribute="sex",
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

## Utility

### IntersectionalBinarizer

Creates a binary protected-group column from complex multi-attribute criteria. Useful when the protected group is defined by the intersection of multiple attributes.

Supports:
- Equality: `{"race": "White"}`
- List membership: `{"race": ["White", "Asian"]}`
- Threshold: `{"age": {">": 65}}`

```python
from skfair.preprocessing import IntersectionalBinarizer

binarizer = IntersectionalBinarizer(
    conditions={"sex": 1, "race": ["White"]},
    output_column="privileged",
)
X_out = binarizer.fit_transform(X)
```

### DropColumns

Drop named columns inside an sklearn Pipeline:

```python
from skfair.preprocessing import DropColumns
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("drop_sens", DropColumns(columns=["sex"])),
    ("clf", LogisticRegression()),
])
```

---

## Pipeline integration

Samplers follow the imbalanced-learn API and work with `imblearn.pipeline.Pipeline`:

```python
from imblearn.pipeline import Pipeline
from skfair.preprocessing import Massaging
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("fair", Massaging(sens_attr="sex", priv_group=1)),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)
```

Transformers (`GeometricFairnessRepair`, `IntersectionalBinarizer`, `DropColumns`) are standard sklearn transformers and work inside a regular `sklearn.pipeline.Pipeline`.

> **Tip:** We recommend always using `imblearn.pipeline.Pipeline` — it extends sklearn's Pipeline with `fit_resample` support, so it works with all scikit-fair methods (transformers, samplers, and meta-estimators) without needing to switch imports.
