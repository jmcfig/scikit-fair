"""
Tests for OptimizedPreprocessing (Calmon et al., 2017).

Uses a small categorical dataset since the algorithm requires discrete features.
"""
import numpy as np
import pandas as pd
import pytest

from skfair.preprocessing import OptimizedPreprocessing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _distortion(old, new):
    """Simple distortion: +1 per changed feature, +2 for changed label."""
    cost = 0.0
    for k in old:
        if k != "label" and old[k] != new[k]:
            cost += 1.0
    if old["label"] != new["label"]:
        cost += 2.0
    return cost


def _make_categorical_data(n_per_cell=10, seed=42):
    """
    Build a categorical dataset with known group imbalance.

    1 categorical feature (age_cat) x 2 groups x 2 labels.
    Group 0 has lower positive rate than group 1.
    Kept small (few feature categories) to keep the optimisation feasible.
    """
    rng = np.random.RandomState(seed)

    age_vals = ["young", "old"]

    rows = []
    labels = []

    # Group 0 (unprivileged): ~35 % positive
    for _ in range(n_per_cell * 2):
        rows.append({
            "age_cat": rng.choice(age_vals),
            "group": 0,
        })
        labels.append(1 if rng.rand() < 0.35 else 0)

    # Group 1 (privileged): ~65 % positive
    for _ in range(n_per_cell * 2):
        rows.append({
            "age_cat": rng.choice(age_vals),
            "group": 1,
        })
        labels.append(1 if rng.rand() < 0.65 else 0)

    X = pd.DataFrame(rows)
    y = np.array(labels)
    return X, y


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def categorical_data():
    return _make_categorical_data(n_per_cell=10)


@pytest.fixture
def larger_categorical_data():
    return _make_categorical_data(n_per_cell=20)


@pytest.fixture
def op_kwargs():
    return dict(
        sens_attr="group",
        features_to_transform=["age_cat"],
        distortion_fun=_distortion,
        epsilon=0.5,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOptimizedPreprocessing:
    """Core tests for OptimizedPreprocessing."""

    def test_fit_resample_returns_correct_types(self, categorical_data, op_kwargs):
        """fit_resample returns DataFrame and ndarray."""
        X, y = categorical_data
        op = OptimizedPreprocessing(**op_kwargs)
        X_out, y_out = op.fit_resample(X, y)

        assert isinstance(X_out, pd.DataFrame)
        assert isinstance(y_out, np.ndarray)

    def test_same_sample_count(self, categorical_data, op_kwargs):
        """Clean-sampling: no samples added or removed."""
        X, y = categorical_data
        op = OptimizedPreprocessing(**op_kwargs)
        X_out, y_out = op.fit_resample(X, y)

        assert len(X_out) == len(X)
        assert len(y_out) == len(y)

    def test_labels_are_binary(self, categorical_data, op_kwargs):
        """Output labels remain binary {0, 1}."""
        X, y = categorical_data
        op = OptimizedPreprocessing(**op_kwargs)
        _, y_out = op.fit_resample(X, y)

        assert set(np.unique(y_out)).issubset({0, 1})

    def test_features_are_from_domain(self, categorical_data, op_kwargs):
        """Transformed feature values stay within the original domain."""
        X, y = categorical_data
        op = OptimizedPreprocessing(**op_kwargs)
        X_out, _ = op.fit_resample(X, y)

        for col in op_kwargs["features_to_transform"]:
            original_domain = set(X[col].unique())
            assert set(X_out[col].unique()).issubset(original_domain)

    def test_columns_preserved(self, categorical_data, op_kwargs):
        """All original columns are present in the output."""
        X, y = categorical_data
        op = OptimizedPreprocessing(**op_kwargs)
        X_out, _ = op.fit_resample(X, y)

        assert list(X_out.columns) == list(X.columns)

    def test_sensitive_attribute_unchanged(self, categorical_data, op_kwargs):
        """Protected attribute column is not modified."""
        X, y = categorical_data
        op = OptimizedPreprocessing(**op_kwargs)
        X_out, _ = op.fit_resample(X, y)

        assert (X_out["group"].values == X["group"].values).all()

    def test_fitted_attributes(self, categorical_data, op_kwargs):
        """mapping_ and classes_ exist after fit_resample."""
        X, y = categorical_data
        op = OptimizedPreprocessing(**op_kwargs)
        op.fit_resample(X, y)

        assert hasattr(op, "mapping_")
        assert hasattr(op, "classes_")
        assert isinstance(op.mapping_, dict)

    def test_reproducibility(self, categorical_data, op_kwargs):
        """Same random_state produces identical results."""
        X, y = categorical_data

        op1 = OptimizedPreprocessing(**op_kwargs)
        X1, y1 = op1.fit_resample(X, y)

        op2 = OptimizedPreprocessing(**op_kwargs)
        X2, y2 = op2.fit_resample(X, y)

        assert X1.equals(X2)
        assert (y1 == y2).all()

    def test_different_seeds(self, categorical_data, op_kwargs):
        """Different random_state may produce different results."""
        X, y = categorical_data

        kw1 = {**op_kwargs, "random_state": 42}
        kw2 = {**op_kwargs, "random_state": 999}

        op1 = OptimizedPreprocessing(**kw1)
        X1, y1 = op1.fit_resample(X, y)

        op2 = OptimizedPreprocessing(**kw2)
        X2, y2 = op2.fit_resample(X, y)

        # With different seeds the randomised mapping should (usually) differ
        differs = not X1.equals(X2) or not (y1 == y2).all()
        assert differs


class TestOptimizedPreprocessingValidation:
    """Input validation tests."""

    def test_missing_features_to_transform(self, categorical_data):
        """Raises when features_to_transform columns are missing from X."""
        X, y = categorical_data
        op = OptimizedPreprocessing(
            sens_attr="group",
            features_to_transform=["nonexistent"],
            distortion_fun=_distortion,
        )
        with pytest.raises(ValueError, match="features_to_transform"):
            op.fit_resample(X, y)

    def test_non_callable_distortion(self, categorical_data):
        """Raises when distortion_fun is not callable."""
        X, y = categorical_data
        op = OptimizedPreprocessing(
            sens_attr="group",
            features_to_transform=["age_cat"],
            distortion_fun="not_callable",
            epsilon=0.5,
        )
        with pytest.raises(TypeError, match="callable"):
            op.fit_resample(X, y)

    def test_non_binary_labels(self, categorical_data):
        """Raises when labels are not binary."""
        X, _ = categorical_data
        y_multi = np.array([0, 1, 2] * (len(X) // 3) + [0] * (len(X) % 3))
        op = OptimizedPreprocessing(
            sens_attr="group",
            features_to_transform=["age_cat"],
            distortion_fun=_distortion,
            epsilon=0.5,
        )
        with pytest.raises(ValueError, match="Binary labels"):
            op.fit_resample(X, y_multi)

    def test_mismatched_clist_dlist(self, categorical_data):
        """Raises when clist and dlist have different lengths."""
        X, y = categorical_data
        op = OptimizedPreprocessing(
            sens_attr="group",
            features_to_transform=["age_cat"],
            distortion_fun=_distortion,
            epsilon=0.5,
            clist=[0.99, 1.99],
            dlist=[0.1],
        )
        with pytest.raises(ValueError, match="same length"):
            op.fit_resample(X, y)

    def test_sens_attr_none(self, categorical_data):
        """Raises when sens_attr is None."""
        X, y = categorical_data
        op = OptimizedPreprocessing(
            sens_attr=None,
            features_to_transform=["age_cat"],
            distortion_fun=_distortion,
            epsilon=0.5,
        )
        with pytest.raises(ValueError, match="sens_attr"):
            op.fit_resample(X, y)
