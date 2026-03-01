import pytest
from sklearn.tree import DecisionTreeClassifier
from skfair.preprocessing import ReweighingClassifier


class TestReweighingClassifier:
    """Tests for ReweighingClassifier meta-estimator."""

    def test_fit_predict_basic(self, simple_binary_data, sens_attr):
        """Basic fit/predict workflow works."""
        X, y = simple_binary_data
        clf = ReweighingClassifier(sens_attr=sens_attr)
        clf.fit(X, y)
        preds = clf.predict(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, simple_binary_data, sens_attr):
        """predict_proba returns correct shape."""
        X, y = simple_binary_data
        clf = ReweighingClassifier(sens_attr=sens_attr)
        clf.fit(X, y)
        proba = clf.predict_proba(X)

        assert proba.shape == (len(X), 2)

    def test_custom_estimator(self, simple_binary_data, sens_attr):
        """Works with custom estimator."""
        X, y = simple_binary_data
        clf = ReweighingClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            sens_attr=sens_attr
        )
        clf.fit(X, y)
        preds = clf.predict(X)

        assert len(preds) == len(X)
