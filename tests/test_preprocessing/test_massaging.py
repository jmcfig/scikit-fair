import pytest
import numpy as np
from skfair.preprocessing import Massaging


class TestMassaging:
    """Tests for Massaging label modification."""

    def test_fit_resample_returns_correct_types(self, simple_binary_data, sens_attr, priv_group):
        """fit_resample returns DataFrame and array."""
        X, y = simple_binary_data
        ms = Massaging(sens_attr=sens_attr, priv_group=priv_group)
        X_out, y_out = ms.fit_resample(X, y)

        assert X_out.shape == X.shape
        assert len(y_out) == len(y)

    def test_labels_are_binary(self, simple_binary_data, sens_attr, priv_group):
        """Output labels remain binary."""
        X, y = simple_binary_data
        ms = Massaging(sens_attr=sens_attr, priv_group=priv_group)
        _, y_out = ms.fit_resample(X, y)

        assert set(np.unique(y_out)).issubset({0, 1})

    def test_same_sample_count(self, simple_binary_data, sens_attr, priv_group):
        """Massaging does not add or remove samples."""
        X, y = simple_binary_data
        ms = Massaging(sens_attr=sens_attr, priv_group=priv_group)
        X_out, y_out = ms.fit_resample(X, y)

        assert len(X_out) == len(X)
        assert len(y_out) == len(y)
