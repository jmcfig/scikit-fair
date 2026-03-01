"""
Tests for LearningFairRepresentations.

Covers: fit / transform basics, sklearn integration (clone, get/set_params,
Pipeline, cross_val_score, GridSearchCV), reproducibility, and edge cases.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.utils.validation import check_is_fitted

from skfair.preprocessing import LearningFairRepresentations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lfr_data():
    """100-sample dataset suited for LFR tests (continuous features)."""
    rng = np.random.RandomState(0)
    n = 50
    # Group 0 – lower positive rate
    X0 = pd.DataFrame({
        "feat1": rng.randn(n),
        "feat2": rng.randn(n),
        "group": np.zeros(n, dtype=int),
    })
    y0 = (rng.rand(n) < 0.3).astype(int)

    # Group 1 – higher positive rate
    X1 = pd.DataFrame({
        "feat1": rng.randn(n) + 0.5,
        "feat2": rng.randn(n) + 0.5,
        "group": np.ones(n, dtype=int),
    })
    y1 = (rng.rand(n) < 0.7).astype(int)

    X = pd.concat([X0, X1], ignore_index=True)
    y = np.concatenate([y0, y1])
    return X, y


@pytest.fixture
def lfr_default(lfr_data):
    """Fitted LFR with default-ish hyper-parameters."""
    X, y = lfr_data
    lfr = LearningFairRepresentations(
        sens_attr="group",
        priv_group=1,
        k=3,
        Ax=0.01,
        Ay=1.0,
        Az=50.0,
        maxiter=500,
        maxfun=500,
        random_state=42,
    )
    lfr.fit(X, y)
    return lfr


# ---------------------------------------------------------------------------
# Basic fit / transform
# ---------------------------------------------------------------------------

class TestFitTransform:
    """Core behaviour of fit() and transform()."""

    def test_fit_returns_self(self, lfr_data):
        X, y = lfr_data
        lfr = LearningFairRepresentations(
            sens_attr="group", priv_group=1,
            k=3, maxiter=100, maxfun=100, random_state=0,
        )
        result = lfr.fit(X, y)
        assert result is lfr

    def test_fitted_attributes_exist(self, lfr_default):
        check_is_fitted(lfr_default)
        assert hasattr(lfr_default, "w_")
        assert hasattr(lfr_default, "prototypes_")
        assert hasattr(lfr_default, "features_dim_")
        assert hasattr(lfr_default, "feature_columns_")

    def test_prototypes_shape(self, lfr_default):
        assert lfr_default.prototypes_.shape == (3, 2)  # k=3, 2 features
        assert lfr_default.w_.shape == (3,)

    def test_feature_columns_exclude_sensitive(self, lfr_default):
        assert "group" not in lfr_default.feature_columns_
        assert set(lfr_default.feature_columns_) == {"feat1", "feat2"}

    def test_transform_output_shape(self, lfr_default, lfr_data):
        X, _ = lfr_data
        X_fair = lfr_default.transform(X)
        assert X_fair.shape == X.shape

    def test_transform_returns_dataframe(self, lfr_default, lfr_data):
        X, _ = lfr_data
        X_fair = lfr_default.transform(X)
        assert isinstance(X_fair, pd.DataFrame)

    def test_transform_preserves_sens_attr(self, lfr_default, lfr_data):
        X, _ = lfr_data
        X_fair = lfr_default.transform(X)
        pd.testing.assert_series_equal(X_fair["group"], X["group"])

    def test_transform_modifies_features(self, lfr_default, lfr_data):
        X, _ = lfr_data
        X_fair = lfr_default.transform(X)
        # Reconstructed features should differ from originals
        assert not np.allclose(X_fair["feat1"].values, X["feat1"].values)

    def test_transform_does_not_mutate_input(self, lfr_default, lfr_data):
        X, _ = lfr_data
        X_orig = X.copy()
        lfr_default.transform(X)
        pd.testing.assert_frame_equal(X, X_orig)

    def test_fit_transform_equivalent(self, lfr_data):
        X, y = lfr_data
        kwargs = dict(
            sens_attr="group", priv_group=1,
            k=3, maxiter=200, maxfun=200, random_state=42,
        )
        lfr1 = LearningFairRepresentations(**kwargs)
        X1 = lfr1.fit(X, y).transform(X)

        lfr2 = LearningFairRepresentations(**kwargs)
        X2 = lfr2.fit_transform(X, y)

        pd.testing.assert_frame_equal(X1, X2)


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------

class TestValidation:
    """Input validation and error paths."""

    def test_fit_rejects_non_dataframe(self):
        lfr = LearningFairRepresentations(
            sens_attr="group", priv_group=1,
        )
        with pytest.raises(TypeError, match="pandas DataFrame"):
            lfr.fit(np.zeros((10, 3)), np.zeros(10))

    def test_transform_rejects_non_dataframe(self, lfr_default):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            lfr_default.transform(np.zeros((10, 3)))

    def test_fit_rejects_missing_sens_attr(self, lfr_data):
        X, y = lfr_data
        lfr = LearningFairRepresentations(
            sens_attr="nonexistent", priv_group=1,
        )
        with pytest.raises(ValueError, match="not found"):
            lfr.fit(X, y)

    def test_transform_before_fit_raises(self, lfr_data):
        X, _ = lfr_data
        lfr = LearningFairRepresentations(
            sens_attr="group", priv_group=1,
        )
        with pytest.raises(Exception):
            # Should raise NotFittedError from check_is_fitted
            lfr.transform(X)


# ---------------------------------------------------------------------------
# Sklearn compatibility
# ---------------------------------------------------------------------------

class TestSklearnCompat:
    """clone, get_params, set_params."""

    def test_clone_unfitted(self):
        lfr = LearningFairRepresentations(
            sens_attr="group", priv_group=1, k=7,
        )
        cloned = clone(lfr)
        assert type(cloned) is type(lfr)
        assert cloned is not lfr
        assert cloned.get_params() == lfr.get_params()

    def test_clone_fitted(self, lfr_default):
        cloned = clone(lfr_default)
        # Cloned should be unfitted
        assert not hasattr(cloned, "w_")
        assert cloned.get_params() == lfr_default.get_params()

    def test_get_params_contains_all_init_args(self):
        lfr = LearningFairRepresentations(
            sens_attr="group", priv_group=1,
        )
        params = lfr.get_params()
        expected = {
            "sens_attr", "priv_group", "k", "Ax", "Ay", "Az",
            "maxiter", "maxfun", "random_state", "verbose",
        }
        assert expected == set(params.keys())

    def test_set_params_roundtrip(self):
        lfr1 = LearningFairRepresentations(
            sens_attr="group", priv_group=1, k=7,
        )
        params = lfr1.get_params()
        lfr2 = LearningFairRepresentations(
            sens_attr="group", priv_group=1,
        )
        lfr2.set_params(**params)
        assert lfr1.get_params() == lfr2.get_params()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Same random_state → identical results."""

    def test_same_seed_same_output(self, lfr_data):
        X, y = lfr_data
        kwargs = dict(
            sens_attr="group", priv_group=1,
            k=3, maxiter=200, maxfun=200, random_state=42,
        )

        lfr1 = LearningFairRepresentations(**kwargs)
        X1 = lfr1.fit_transform(X, y)

        lfr2 = LearningFairRepresentations(**kwargs)
        X2 = lfr2.fit_transform(X, y)

        pd.testing.assert_frame_equal(X1, X2)
        np.testing.assert_array_equal(lfr1.w_, lfr2.w_)
        np.testing.assert_array_equal(lfr1.prototypes_, lfr2.prototypes_)

    def test_different_seed_different_output(self, lfr_data):
        X, y = lfr_data
        base = dict(
            sens_attr="group", priv_group=1,
            k=3, maxiter=200, maxfun=200,
        )

        lfr1 = LearningFairRepresentations(**base, random_state=0)
        X1 = lfr1.fit_transform(X, y)

        lfr2 = LearningFairRepresentations(**base, random_state=99)
        X2 = lfr2.fit_transform(X, y)

        # Different initialisations → (very likely) different results
        assert not np.allclose(X1["feat1"].values, X2["feat1"].values)


# ---------------------------------------------------------------------------
# Pipeline integration (the key part)
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """LFR in sklearn Pipelines, cross-validation, grid search."""

    def test_pipeline_fit_predict(self, lfr_data):
        X, y = lfr_data
        pipe = SklearnPipeline([
            ("lfr", LearningFairRepresentations(
                sens_attr="group", priv_group=1,
                k=3, maxiter=300, maxfun=300, random_state=42,
            )),
            ("clf", LogisticRegression(solver="liblinear", max_iter=200)),
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_pipeline_predict_proba(self, lfr_data):
        X, y = lfr_data
        pipe = SklearnPipeline([
            ("lfr", LearningFairRepresentations(
                sens_attr="group", priv_group=1,
                k=3, maxiter=300, maxfun=300, random_state=42,
            )),
            ("clf", LogisticRegression(solver="liblinear", max_iter=200)),
        ])

        pipe.fit(X, y)
        proba = pipe.predict_proba(X)

        assert proba.shape == (len(X), 2)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

    def test_pipeline_score(self, lfr_data):
        X, y = lfr_data
        pipe = SklearnPipeline([
            ("lfr", LearningFairRepresentations(
                sens_attr="group", priv_group=1,
                k=3, maxiter=300, maxfun=300, random_state=42,
            )),
            ("clf", LogisticRegression(solver="liblinear", max_iter=200)),
        ])
        pipe.fit(X, y)
        score = pipe.score(X, y)
        assert 0 <= score <= 1

    def test_pipeline_with_tree_classifier(self, lfr_data):
        X, y = lfr_data
        pipe = SklearnPipeline([
            ("lfr", LearningFairRepresentations(
                sens_attr="group", priv_group=1,
                k=3, maxiter=300, maxfun=300, random_state=42,
            )),
            ("clf", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(X)

    def test_cross_validation(self, lfr_data):
        X, y = lfr_data
        pipe = SklearnPipeline([
            ("lfr", LearningFairRepresentations(
                sens_attr="group", priv_group=1,
                k=3, maxiter=300, maxfun=300, random_state=42,
            )),
            ("clf", LogisticRegression(solver="liblinear", max_iter=200)),
        ])

        scores = cross_val_score(pipe, X, y, cv=3, scoring="accuracy")
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_grid_search(self, lfr_data):
        X, y = lfr_data
        pipe = SklearnPipeline([
            ("lfr", LearningFairRepresentations(
                sens_attr="group", priv_group=1,
                maxiter=200, maxfun=200, random_state=42,
            )),
            ("clf", LogisticRegression(solver="liblinear", max_iter=200)),
        ])

        param_grid = {"lfr__k": [2, 4]}
        gs = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy")
        gs.fit(X, y)

        assert gs.best_params_ is not None
        assert 0 <= gs.best_score_ <= 1

    def test_pipeline_reproducibility(self, lfr_data):
        X, y = lfr_data

        def make_pipe():
            return SklearnPipeline([
                ("lfr", LearningFairRepresentations(
                    sens_attr="group", priv_group=1,
                    k=3, maxiter=300, maxfun=300, random_state=42,
                )),
                ("clf", LogisticRegression(
                    solver="liblinear", max_iter=200, random_state=0,
                )),
            ])

        pipe1 = make_pipe()
        pipe1.fit(X, y)
        preds1 = pipe1.predict(X)

        pipe2 = make_pipe()
        pipe2.fit(X, y)
        preds2 = pipe2.predict(X)

        np.testing.assert_array_equal(preds1, preds2)


# ---------------------------------------------------------------------------
# Hyperparameter sensitivity
# ---------------------------------------------------------------------------

class TestHyperparameters:
    """Different k / weight values produce valid outputs."""

    @pytest.mark.parametrize("k", [2, 5, 10])
    def test_varying_k(self, lfr_data, k):
        X, y = lfr_data
        lfr = LearningFairRepresentations(
            sens_attr="group", priv_group=1,
            k=k, maxiter=200, maxfun=200, random_state=42,
        )
        X_fair = lfr.fit_transform(X, y)
        assert X_fair.shape == X.shape
        assert lfr.prototypes_.shape == (k, 2)

    @pytest.mark.parametrize("Az", [0.0, 10.0, 100.0])
    def test_varying_fairness_weight(self, lfr_data, Az):
        X, y = lfr_data
        lfr = LearningFairRepresentations(
            sens_attr="group", priv_group=1,
            k=3, Az=Az, maxiter=200, maxfun=200, random_state=42,
        )
        X_fair = lfr.fit_transform(X, y)
        assert X_fair.shape == X.shape
