import pytest
import pandas as pd
from skfair.preprocessing import FAWOS


class TestFAWOS:
    """Tests for FAWOS oversampler."""

    def test_fit_resample_runs(self, simple_binary_data, sens_attr, priv_group):
        """Basic smoke test for fit_resample."""
        X, y = simple_binary_data
        fawos = FAWOS(sens_attr=sens_attr, priv_group=priv_group, random_state=42)
        X_out, y_out = fawos.fit_resample(X, y)

        assert len(X_out) >= len(X)
        assert len(y_out) == len(X_out)

    def test_output_is_dataframe(self, simple_binary_data, sens_attr, priv_group):
        """Output X should be a DataFrame."""
        X, y = simple_binary_data
        fawos = FAWOS(sens_attr=sens_attr, priv_group=priv_group, random_state=42)
        X_out, _ = fawos.fit_resample(X, y)

        assert isinstance(X_out, pd.DataFrame)
        assert X_out.columns.tolist() == X.columns.tolist()

    def test_pos_label_flips_direction(self, simple_binary_data, sens_attr, priv_group):
        """pos_label=0 oversamples the y=0 class of the unprivileged group."""
        X, y = simple_binary_data
        fawos = FAWOS(sens_attr=sens_attr, priv_group=priv_group,
                      pos_label=0, random_state=42)
        X_out, y_out = fawos.fit_resample(X, y)

        assert len(y_out) >= len(y)
        # every synthetic sample carries the requested favourable label
        assert (pd.Series(y_out).iloc[len(y):] == 0).all()

    def test_pos_label_unknown_raises(self, simple_binary_data, sens_attr, priv_group):
        """An unknown pos_label raises a clear error."""
        X, y = simple_binary_data
        fawos = FAWOS(sens_attr=sens_attr, priv_group=priv_group, pos_label=7)
        with pytest.raises(ValueError, match="pos_label"):
            fawos.fit_resample(X, y)
