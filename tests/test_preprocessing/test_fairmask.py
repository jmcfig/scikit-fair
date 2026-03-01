import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from skfair.preprocessing import FairMask


class TestFairMask:
    """Tests for FairMask meta-estimator."""

    def test_fit_predict_basic(self, simple_binary_data, sens_attr):
        """Basic fit/predict workflow works."""
        X, y = simple_binary_data
        clf = FairMask(sens_attr=sens_attr, budget=3)
        clf.fit(X, y)
        preds = clf.predict(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, simple_binary_data, sens_attr):
        """predict_proba returns correct shape."""
        X, y = simple_binary_data
        clf = FairMask(sens_attr=sens_attr, budget=3)
        clf.fit(X, y)
        proba = clf.predict_proba(X)

        assert proba.shape == (len(X), 2)

    def test_custom_estimator(self, simple_binary_data, sens_attr):
        """Works with custom estimator."""
        X, y = simple_binary_data
        clf = FairMask(
            estimator=DecisionTreeClassifier(max_depth=2),
            sens_attr=sens_attr,
            budget=3
        )
        clf.fit(X, y)
        preds = clf.predict(X)

        assert len(preds) == len(X)

    def test_custom_extrapolation_model(self, simple_binary_data, sens_attr):
        """Works with custom extrapolation model."""
        X, y = simple_binary_data
        clf = FairMask(
            sens_attr=sens_attr,
            budget=3,
            extrapolation_model=DecisionTreeClassifier(max_depth=2)
        )
        clf.fit(X, y)
        preds = clf.predict(X)

        assert len(preds) == len(X)

    def test_budget_creates_correct_number_of_models(self, simple_binary_data, sens_attr):
        """Budget parameter creates correct number of extrapolation models."""
        X, y = simple_binary_data

        for budget in [1, 5, 10]:
            clf = FairMask(sens_attr=sens_attr, budget=budget)
            clf.fit(X, y)

            assert len(clf.extrapolation_models_) == budget
            assert len(clf.model_weights_) == budget

    def test_model_weights_normalized(self, simple_binary_data, sens_attr):
        """Model weights are normalized to sum to 1."""
        X, y = simple_binary_data
        clf = FairMask(sens_attr=sens_attr, budget=5)
        clf.fit(X, y)

        assert np.isclose(clf.model_weights_.sum(), 1.0)

    def test_sensitive_attr_synthesized(self, simple_binary_data, sens_attr):
        """Protected attribute is synthesized at inference time."""
        X, y = simple_binary_data
        clf = FairMask(sens_attr=sens_attr, budget=5)
        clf.fit(X, y)

        # Access internal method to verify synthesis
        X_masked = clf._synthesize_sensitive_attr(X)

        # Masked X should have same shape
        assert X_masked.shape == X.shape
        # Protected attr column should exist
        assert sens_attr in X_masked.columns
        # Values should be binary
        assert set(X_masked[sens_attr].unique()).issubset({0, 1})

    def test_score_method(self, simple_binary_data, sens_attr):
        """Score method works correctly."""
        X, y = simple_binary_data
        clf = FairMask(sens_attr=sens_attr, budget=3)
        clf.fit(X, y)
        score = clf.score(X, y)

        assert 0.0 <= score <= 1.0

    def test_random_state_reproducibility(self, simple_binary_data, sens_attr):
        """Random state ensures reproducible results."""
        X, y = simple_binary_data

        clf1 = FairMask(sens_attr=sens_attr, budget=5, random_state=42)
        clf1.fit(X, y)
        preds1 = clf1.predict(X)

        clf2 = FairMask(sens_attr=sens_attr, budget=5, random_state=42)
        clf2.fit(X, y)
        preds2 = clf2.predict(X)

        np.testing.assert_array_equal(preds1, preds2)

    def test_missing_sens_attr_raises(self, simple_binary_data):
        """Missing sens_attr raises ValueError."""
        X, y = simple_binary_data
        clf = FairMask(sens_attr=None)

        with pytest.raises(ValueError, match="sens_attr must be specified"):
            clf.fit(X, y)

    def test_invalid_sens_attr_raises(self, simple_binary_data):
        """Invalid sens_attr column raises ValueError."""
        X, y = simple_binary_data
        clf = FairMask(sens_attr='nonexistent_column')

        with pytest.raises(ValueError, match="not found in X"):
            clf.fit(X, y)

    def test_classes_attribute(self, simple_binary_data, sens_attr):
        """Classes attribute is set after fit."""
        X, y = simple_binary_data
        clf = FairMask(sens_attr=sens_attr, budget=3)
        clf.fit(X, y)

        assert hasattr(clf, 'classes_')
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))

    def test_larger_dataset(self, larger_binary_data, sens_attr):
        """Works on larger dataset."""
        X, y = larger_binary_data
        clf = FairMask(
            estimator=RandomForestClassifier(n_estimators=10, random_state=42),
            sens_attr=sens_attr,
            budget=5,
            random_state=42
        )
        clf.fit(X, y)
        preds = clf.predict(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})
