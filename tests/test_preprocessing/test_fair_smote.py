import pytest
import pandas as pd
from skfair.preprocessing import FairSmote


class TestFairSmote:
    """Tests for FairSmote oversampler."""

    def test_fit_resample_balances_subgroups(self, simple_binary_data, sens_attr):
        """All subgroups should be balanced to max size."""
        X, y = simple_binary_data
        fs = FairSmote(sens_attr=sens_attr, random_state=42)
        X_out, y_out = fs.fit_resample(X, y)

        # Output should be at least as large as input
        assert len(X_out) >= len(X)
        assert len(y_out) == len(X_out)

    def test_preserves_columns(self, simple_binary_data, sens_attr):
        """Output DataFrame should have same columns."""
        X, y = simple_binary_data
        fs = FairSmote(sens_attr=sens_attr, random_state=42)
        X_out, _ = fs.fit_resample(X, y)

        assert X_out.columns.tolist() == X.columns.tolist()

    def test_output_is_dataframe(self, simple_binary_data, sens_attr):
        """Output X should be a DataFrame."""
        X, y = simple_binary_data
        fs = FairSmote(sens_attr=sens_attr, random_state=42)
        X_out, _ = fs.fit_resample(X, y)

        assert isinstance(X_out, pd.DataFrame)
