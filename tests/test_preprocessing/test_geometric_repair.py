import pytest
import numpy as np
from skfair.preprocessing import GeometricFairnessRepair


class TestGeometricFairnessRepair:
    """Tests for GeometricFairnessRepair feature repair."""

    def test_fit_transform_shape(self, simple_binary_data):
        """fit_transform preserves DataFrame shape."""
        X, _ = simple_binary_data
        repair = GeometricFairnessRepair(
            sens_attr='group',
            repair_columns=['age', 'income']
        )
        X_out = repair.fit_transform(X)

        assert X_out.shape == X.shape

    def test_lambda_zero_no_change(self, simple_binary_data):
        """lambda=0 should not modify features."""
        X, _ = simple_binary_data
        repair = GeometricFairnessRepair(
            sens_attr='group',
            repair_columns=['age', 'income'],
            lambda_param=0.0
        )
        X_out = repair.fit_transform(X)

        np.testing.assert_array_almost_equal(X_out['age'].values, X['age'].values)

    def test_lambda_one_modifies_features(self, simple_binary_data):
        """lambda=1 should modify features."""
        X, _ = simple_binary_data
        repair = GeometricFairnessRepair(
            sens_attr='group',
            repair_columns=['age'],
            lambda_param=1.0
        )
        X_out = repair.fit_transform(X)

        # At least some values should change
        assert not np.allclose(X_out['age'].values, X['age'].values)
