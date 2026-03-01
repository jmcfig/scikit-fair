import pytest
import pandas as pd
from skfair.preprocessing import FairOversampling


class TestFairOversampling:
    """Tests for FairOversampling."""

    def test_fit_resample_increases_samples(self, simple_binary_data, sens_attr, priv_group):
        """Oversampling should increase or maintain sample count."""
        X, y = simple_binary_data
        fos = FairOversampling(sens_attr=sens_attr, priv_group=priv_group, random_state=42)
        X_out, y_out = fos.fit_resample(X, y)

        assert len(X_out) >= len(X)
        assert len(y_out) == len(X_out)

    def test_output_types(self, simple_binary_data, sens_attr, priv_group):
        """Output should be DataFrame and array."""
        X, y = simple_binary_data
        fos = FairOversampling(sens_attr=sens_attr, priv_group=priv_group, random_state=42)
        X_out, y_out = fos.fit_resample(X, y)

        assert isinstance(X_out, pd.DataFrame)
        assert X_out.columns.tolist() == X.columns.tolist()
