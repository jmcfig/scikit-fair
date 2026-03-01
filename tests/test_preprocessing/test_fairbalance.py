import pytest
import numpy as np
import pandas as pd
from skfair.preprocessing import FairBalance


class TestFairBalance:
    """Tests for FairBalance preprocessor."""

    def test_fit_returns_self(self, simple_binary_data, sens_attr):
        """fit() should return self for method chaining."""
        X, y = simple_binary_data
        fb = FairBalance(sens_attr=sens_attr)
        result = fb.fit(X, y)
        assert result is fb

    def test_fit_transform_returns_tuple(self, simple_binary_data, sens_attr):
        """fit_transform() returns (X, weights) tuple."""
        X, y = simple_binary_data
        fb = FairBalance(sens_attr=sens_attr)
        X_out, weights = fb.fit_transform(X, y)

        assert X_out.shape == X.shape
        assert len(weights) == len(X)

    def test_weights_are_positive(self, simple_binary_data, sens_attr):
        """All weights should be positive."""
        X, y = simple_binary_data
        fb = FairBalance(sens_attr=sens_attr)
        _, weights = fb.fit_transform(X, y)

        assert (weights > 0).all()

    def test_weights_balance_class_distribution(self, simple_binary_data, sens_attr):
        """
        Weighted class distribution should be 1:1 within each group.

        For each group a, sum of weights for Y=0 should equal sum for Y=1.
        """
        X, y = simple_binary_data
        fb = FairBalance(sens_attr=sens_attr)
        _, weights = fb.fit_transform(X, y)

        y_series = pd.Series(y, index=X.index)

        for group_val in X[sens_attr].unique():
            mask = X[sens_attr] == group_val
            w_pos = weights[mask & (y_series == 1)].sum()
            w_neg = weights[mask & (y_series == 0)].sum()
            # Weighted counts should be equal (balanced 1:1)
            assert np.isclose(w_pos, w_neg), (
                f"Group {group_val}: weighted pos={w_pos}, neg={w_neg}"
            )

    def test_weight_formula(self, simple_binary_data, sens_attr):
        """Verify the weight formula: w = |A=a| / |A=a, Y=y|."""
        X, y = simple_binary_data
        fb = FairBalance(sens_attr=sens_attr)
        _, weights = fb.fit_transform(X, y)

        y_series = pd.Series(y, index=X.index)
        A = X[sens_attr]

        for idx in X.index:
            a_i = A.loc[idx]
            y_i = y_series.loc[idx]
            n_a = (A == a_i).sum()
            n_a_y = ((A == a_i) & (y_series == y_i)).sum()
            expected_w = n_a / n_a_y
            assert np.isclose(weights.loc[idx], expected_w)

    def test_variant_mode(self, simple_binary_data, sens_attr):
        """Variant mode should use different formula and rescale."""
        X, y = simple_binary_data

        fb_normal = FairBalance(sens_attr=sens_attr, variant=False)
        _, w_normal = fb_normal.fit_transform(X, y)

        fb_variant = FairBalance(sens_attr=sens_attr, variant=True)
        _, w_variant = fb_variant.fit_transform(X, y)

        # Weights should be different
        assert not np.allclose(w_normal, w_variant)

        # Variant weights should sum to n_samples
        assert np.isclose(w_variant.sum(), len(X))

    def test_variant_weight_formula(self, simple_binary_data, sens_attr):
        """Verify variant formula: w = 1 / |A=a, Y=y| (before rescaling)."""
        X, y = simple_binary_data
        fb = FairBalance(sens_attr=sens_attr, variant=True)
        _, weights = fb.fit_transform(X, y)

        y_series = pd.Series(y, index=X.index)
        A = X[sens_attr]
        n = len(X)

        # Compute expected weights before rescaling
        raw_weights = []
        for idx in X.index:
            a_i = A.loc[idx]
            y_i = y_series.loc[idx]
            n_a_y = ((A == a_i) & (y_series == y_i)).sum()
            raw_weights.append(1.0 / n_a_y)

        raw_weights = np.array(raw_weights)
        scale = n / raw_weights.sum()
        expected = raw_weights * scale

        assert np.allclose(weights.values, expected)

    def test_requires_dataframe(self, simple_binary_data, sens_attr):
        """Should raise TypeError if X is not a DataFrame."""
        X, y = simple_binary_data
        fb = FairBalance(sens_attr=sens_attr)

        with pytest.raises(TypeError, match="pandas DataFrame"):
            fb.fit_transform(X.values, y)

    def test_requires_sens_attr(self, simple_binary_data):
        """Should raise ValueError if sens_attr is not set."""
        X, y = simple_binary_data
        fb = FairBalance(sens_attr=None)

        with pytest.raises(ValueError, match="sens_attr must be set"):
            fb.fit_transform(X, y)

    def test_sens_attr_must_exist(self, simple_binary_data):
        """Should raise ValueError if sens_attr not in columns."""
        X, y = simple_binary_data
        fb = FairBalance(sens_attr="nonexistent")

        with pytest.raises(ValueError, match="not found in X columns"):
            fb.fit_transform(X, y)

    def test_requires_binary_labels(self, simple_binary_data, sens_attr):
        """Should raise ValueError for non-binary labels."""
        X, y = simple_binary_data
        y_multiclass = np.array([0, 1, 2] * (len(y) // 3) + [0] * (len(y) % 3))
        fb = FairBalance(sens_attr=sens_attr)

        with pytest.raises(ValueError, match="binary labels"):
            fb.fit_transform(X, y_multiclass)

    def test_pos_label_must_exist(self, simple_binary_data, sens_attr):
        """Should raise ValueError if pos_label not in labels."""
        X, y = simple_binary_data
        fb = FairBalance(sens_attr=sens_attr, pos_label=99)

        with pytest.raises(ValueError, match="pos_label=99"):
            fb.fit_transform(X, y)

    def test_stores_fitted_attributes(self, simple_binary_data, sens_attr):
        """After fit_transform, should have fitted attributes."""
        X, y = simple_binary_data
        fb = FairBalance(sens_attr=sens_attr)
        fb.fit_transform(X, y)

        assert hasattr(fb, 'classes_')
        assert hasattr(fb, 'weight_table_')
        assert hasattr(fb, 'weights_')
        assert hasattr(fb, 'group_counts_')
        assert hasattr(fb, 'joint_counts_')

    def test_x_unchanged(self, simple_binary_data, sens_attr):
        """fit_transform should not modify X."""
        X, y = simple_binary_data
        X_original = X.copy()
        fb = FairBalance(sens_attr=sens_attr)
        X_out, _ = fb.fit_transform(X, y)

        pd.testing.assert_frame_equal(X_out, X_original)

    def test_reproducibility(self, simple_binary_data, sens_attr):
        """Same input should produce same weights."""
        X, y = simple_binary_data

        fb1 = FairBalance(sens_attr=sens_attr)
        _, w1 = fb1.fit_transform(X, y)

        fb2 = FairBalance(sens_attr=sens_attr)
        _, w2 = fb2.fit_transform(X, y)

        pd.testing.assert_series_equal(w1, w2)

    def test_works_with_larger_data(self, larger_binary_data, sens_attr):
        """Should work with larger datasets."""
        X, y = larger_binary_data
        fb = FairBalance(sens_attr=sens_attr)
        X_out, weights = fb.fit_transform(X, y)

        assert len(weights) == len(X)
        assert (weights > 0).all()
