import pytest
import pandas as pd
from skfair.preprocessing import FairwayRemover


class TestFairwayRemover:
    """Tests for FairwayRemover clean sampler."""

    def test_fit_resample_removes_or_keeps_samples(self, simple_binary_data, sens_attr, priv_group):
        """Fairway should remove or maintain sample count (not add)."""
        X, y = simple_binary_data
        fw = FairwayRemover(sens_attr=sens_attr, priv_group=priv_group)
        X_out, y_out = fw.fit_resample(X, y)

        # Should not add samples
        assert len(X_out) <= len(X)
        assert len(y_out) == len(X_out)

    def test_output_is_dataframe(self, simple_binary_data, sens_attr, priv_group):
        """Output X should be a DataFrame."""
        X, y = simple_binary_data
        fw = FairwayRemover(sens_attr=sens_attr, priv_group=priv_group)
        X_out, _ = fw.fit_resample(X, y)

        assert isinstance(X_out, pd.DataFrame)

    def test_models_fitted(self, simple_binary_data, sens_attr, priv_group):
        """Privileged and unprivileged models should be fitted."""
        X, y = simple_binary_data
        fw = FairwayRemover(sens_attr=sens_attr, priv_group=priv_group)
        fw.fit_resample(X, y)

        assert hasattr(fw, 'model_p_')
        assert hasattr(fw, 'model_u_')
