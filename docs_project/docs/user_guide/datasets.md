# Datasets

`skfair.datasets` provides loaders for six standard fairness benchmark datasets.

All loaders follow the sklearn convention:

```python
load_*(return_X_y=False, as_frame=False)
```

!!! note "Bundled vs. fetched datasets"
    All datasets except ACSIncome ship bundled with the package as CSV files,
    so they load offline with no external dependency. ACSIncome (1.66M rows)
    is too large to bundle: `fetch_acs_income` downloads it once from OpenML
    and caches it locally, following scikit-learn's `fetch_*` convention.

---

## Adult (Census Income)

The Adult dataset from the UCI Machine Learning Repository. Contains 48,842 instances with 14 features. The task is to predict whether income exceeds $50K/yr.

**Common sensitive attribute**: `sex` (1 = male, 0 = female)

```python
from skfair.datasets import load_adult

# Returns a Bunch object
data = load_adult()
X, y = data.data, data.target

# sklearn-style
X, y = load_adult(return_X_y=True, as_frame=True)
print(X.columns.tolist())
print(X["sex"].value_counts())
```

---

## ACSIncome (American Community Survey)

The ACSIncome dataset (Ding et al., 2021), introduced as a large-scale
alternative to Adult, compiled from the American Community Survey (ACS)
Public Use Microdata Sample. The loader fetches the 2018 1-year release for
all US states and Puerto Rico: 1,664,500 instances with 10 features. The
task is to predict whether income exceeds $50K/yr.

**Common sensitive attributes**: `SEX` (1 = male, 0 = female), `RAC1P`
(multi-valued race codes -- pairs of groups can be compared via the
`priv_group`/`unpriv_group` metric arguments)

```python
from skfair.datasets import fetch_acs_income

# Full dataset (downloads once from OpenML, then cached locally)
X, y = fetch_acs_income()

# Tractable subset for demos
X, y = fetch_acs_income(subsample=100_000, random_state=42)
print(X["RAC1P"].value_counts())
```

---

## German Credit

The Statlog (German Credit Data) dataset. Contains 1,000 instances. The task is to predict credit risk (good / bad).

**Common sensitive attribute**: `age` (binarised at a threshold, e.g., >= 25)

```python
from skfair.datasets import load_german

X, y = load_german(return_X_y=True, as_frame=True)
```

---

## Heart Disease

The Cleveland Heart Disease dataset. Contains 303 instances. The task is to predict presence of heart disease.

**Common sensitive attribute**: `age` or `sex`

```python
from skfair.datasets import load_heart_disease

X, y = load_heart_disease(return_X_y=True, as_frame=True)
```

---

## COMPAS (Recidivism)

The ProPublica COMPAS dataset. Contains approximately 7,214 instances with 11 features. The task is to predict two-year recidivism.

**Common sensitive attributes**: `sex` (1 = Male), `race` (1 = Caucasian)

```python
from skfair.datasets import load_compas

X, y = load_compas()
print(X.columns.tolist())
print(X["race"].value_counts())
```

---

## Ricci (Firefighter Promotions)

The Ricci v. DeStefano dataset. Contains 118 instances with 5 features. The task is to predict promotion eligibility based on a binarized combined test score (>= 70).

**Common sensitive attribute**: `Race` (1 = White, 0 = otherwise)

```python
from skfair.datasets import load_ricci

X, y = load_ricci()
print(X.columns.tolist())
print(X["Race"].value_counts())
```

---

## Using datasets in experiments

```python
from skfair.datasets import load_adult
from skfair.preprocessing import Massaging
from skfair.metrics import disparate_impact
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = load_adult(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sampler = Massaging(sens_attr="sex", priv_group=1)
X_fair, y_fair = sampler.fit_resample(X_train, y_train)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_fair, y_fair)
y_pred = clf.predict(X_test)

print(disparate_impact(y_test.values, y_pred, X_test["sex"].values))
```
