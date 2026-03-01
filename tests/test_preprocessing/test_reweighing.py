import pytest
from skfair.preprocessing import Reweighing


class TestReweighing:
    """Tests for Reweighing preprocessor."""

    def test_fit_returns_self(self, simple_binary_data, sens_attr):
        """fit() should return self for method chaining."""
        X, y = simple_binary_data
        rw = Reweighing(sens_attr=sens_attr)
        result = rw.fit(X, y)
        assert result is rw

    def test_fit_transform_returns_tuple(self, simple_binary_data, sens_attr):
        """fit_transform() returns (X, weights) tuple."""
        X, y = simple_binary_data
        rw = Reweighing(sens_attr=sens_attr)
        X_out, weights = rw.fit_transform(X, y)

        assert X_out.shape == X.shape
        assert len(weights) == len(X)

    def test_weights_are_positive(self, simple_binary_data, sens_attr):
        """All weights should be positive."""
        X, y = simple_binary_data
        rw = Reweighing(sens_attr=sens_attr)
        _, weights = rw.fit_transform(X, y)

        assert (weights > 0).all()
