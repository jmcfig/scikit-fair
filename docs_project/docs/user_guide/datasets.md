# Datasets

`skfair.datasets` provides loaders for three standard fairness benchmark datasets.

All loaders follow the sklearn convention:

```python
load_*(return_X_y=False, as_frame=False)
```

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
