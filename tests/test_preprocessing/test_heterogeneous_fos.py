import pytest
import pandas as pd
from skfair.preprocessing import HeterogeneousFOS


class TestHeterogeneousFOS:
    """Tests for HeterogeneousFOS oversampler."""

    def test_fit_resample_runs(self, simple_binary_data, sens_attr):
        """Basic smoke test."""
        X, y = simple_binary_data
        hfos = HeterogeneousFOS(sens_attr=sens_attr, random_state=42)
        X_out, y_out = hfos.fit_resample(X, y)

        assert len(X_out) >= len(X)
        assert len(y_out) == len(X_out)

    def test_output_is_dataframe(self, simple_binary_data, sens_attr):
        """Output X should be a DataFrame."""
        X, y = simple_binary_data
        hfos = HeterogeneousFOS(sens_attr=sens_attr, random_state=42)
        X_out, _ = hfos.fit_resample(X, y)

        assert isinstance(X_out, pd.DataFrame)
        assert X_out.columns.tolist() == X.columns.tolist()
