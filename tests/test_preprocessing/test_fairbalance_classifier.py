import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from skfair.preprocessing import FairBalanceClassifier


class TestFairBalanceClassifier:
    """Tests for FairBalanceClassifier meta-estimator."""

    def test_fit_predict_basic(self, simple_binary_data, sens_attr):
        """Basic fit/predict workflow works."""
        X, y = simple_binary_data
        clf = FairBalanceClassifier(sens_attr=sens_attr)
        clf.fit(X, y)
        preds = clf.predict(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, simple_binary_data, sens_attr):
        """predict_proba returns correct shape."""
        X, y = simple_binary_data
        clf = FairBalanceClassifier(sens_attr=sens_attr)
        clf.fit(X, y)
        proba = clf.predict_proba(X)

        assert proba.shape == (len(X), 2)

    def test_custom_estimator(self, simple_binary_data, sens_attr):
        """Works with custom estimator."""
        X, y = simple_binary_data
        clf = FairBalanceClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            sens_attr=sens_attr
        )
        clf.fit(X, y)
        preds = clf.predict(X)

        assert len(preds) == len(X)

    def test_variant_mode(self, simple_binary_data, sens_attr):
        """Works with variant=True."""
        X, y = simple_binary_data
        clf = FairBalanceClassifier(
            sens_attr=sens_attr,
            variant=True
        )
        clf.fit(X, y)
        preds = clf.predict(X)

        assert len(preds) == len(X)
        # Verify variant weights sum to n
        assert np.isclose(clf.weights_.sum(), len(X))

    def test_score_method(self, simple_binary_data, sens_attr):
        """score() method works."""
        X, y = simple_binary_data
        clf = FairBalanceClassifier(sens_attr=sens_attr)
        clf.fit(X, y)
        score = clf.score(X, y)

        assert 0 <= score <= 1

    def test_stores_fitted_attributes(self, simple_binary_data, sens_attr):
        """After fit, should have fitted attributes."""
        X, y = simple_binary_data
        clf = FairBalanceClassifier(sens_attr=sens_attr)
        clf.fit(X, y)

        assert hasattr(clf, 'fairbalance_')
        assert hasattr(clf, 'estimator_')
        assert hasattr(clf, 'classes_')
        assert hasattr(clf, 'weights_')

    def test_predict_before_fit_raises(self, simple_binary_data, sens_attr):
        """predict() before fit should raise."""
        X, y = simple_binary_data
        clf = FairBalanceClassifier(sens_attr=sens_attr)

        with pytest.raises(Exception):  # NotFittedError
            clf.predict(X)

    def test_works_with_random_forest(self, larger_binary_data, sens_attr):
        """Works with RandomForest estimator."""
        X, y = larger_binary_data
        clf = FairBalanceClassifier(
            estimator=RandomForestClassifier(n_estimators=10, random_state=42),
            sens_attr=sens_attr
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        proba = clf.predict_proba(X)

        assert len(preds) == len(X)
        assert proba.shape == (len(X), 2)

    def test_default_estimator_is_logistic_regression(self, simple_binary_data, sens_attr):
        """Default estimator should be LogisticRegression."""
        X, y = simple_binary_data
        clf = FairBalanceClassifier(sens_attr=sens_attr)
        clf.fit(X, y)

        from sklearn.linear_model import LogisticRegression
        assert isinstance(clf.estimator_, LogisticRegression)

    def test_pos_label_parameter(self, simple_binary_data, sens_attr):
        """pos_label parameter is passed to FairBalance."""
        X, y = simple_binary_data
        clf = FairBalanceClassifier(sens_attr=sens_attr, pos_label=1)
        clf.fit(X, y)

        assert clf.fairbalance_.pos_label == 1
