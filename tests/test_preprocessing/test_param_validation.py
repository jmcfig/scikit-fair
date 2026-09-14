"""Constructor parameter validation of the fair samplers (_parameter_constraints)."""

import numpy as np
import pandas as pd
import pytest

from skfair.preprocessing import (
    FAWOS,
    FairOversampling,
    FairSmote,
    FairwayRemover,
    HeterogeneousFOS,
    Massaging,
    OptimizedPreprocessing,
)


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    n = 200
    X = pd.DataFrame({
        "a": rng.randn(n),
        "b": rng.randn(n),
        "sex": rng.randint(0, 2, n),
    })
    y = (rng.rand(n) < np.where(X["sex"] == 1, 0.6, 0.3)).astype(int)
    return X, y


VALID = [
    (FAWOS, {"priv_group": 1}),
    (FairSmote, {}),
    (FairOversampling, {"priv_group": 1}),
    (HeterogeneousFOS, {}),
    (Massaging, {"priv_group": 1}),
    (FairwayRemover, {"priv_group": 1}),
]


@pytest.mark.parametrize("cls, kwargs", VALID)
def test_valid_params_resample(cls, kwargs, data):
    """Default parameters pass validation (no AttributeError on imblearn >= 0.13)."""
    X, y = data
    X_res, y_res = cls(sens_attr="sex", **kwargs).fit_resample(X, y)
    assert len(X_res) == len(y_res)


INVALID = [
    (FAWOS, {"priv_group": 1, "alpha": -1}, "alpha"),
    (FAWOS, {"priv_group": 1, "rare_weight": -0.5}, "rare_weight"),
    (FAWOS, {"priv_group": 1, "random_state": "abc"}, "random_state"),
    (FairSmote, {"cr": 1.5}, "cr"),
    (FairSmote, {"k_neighbors": 0}, "k_neighbors"),
    (FairOversampling, {"priv_group": 1, "k_neighbors": 0}, "k_neighbors"),
    (HeterogeneousFOS, {"k_neighbors": 2.5}, "k_neighbors"),
    (Massaging, {"estimator": "logreg"}, "estimator"),
    (FairwayRemover, {"priv_group": 1, "estimator": "logreg"}, "estimator"),
    (OptimizedPreprocessing, {"features_to_transform": ["a"], "epsilon": -0.1}, "epsilon"),
]


@pytest.mark.parametrize("cls, kwargs, param", INVALID)
def test_invalid_params_raise(cls, kwargs, param, data):
    """Invalid constructor parameters raise a clear error naming the parameter."""
    X, y = data
    with pytest.raises(ValueError, match=f"'{param}' parameter"):
        cls(sens_attr="sex", **kwargs).fit_resample(X, y)
