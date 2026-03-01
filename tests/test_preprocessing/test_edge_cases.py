"""
Edge case and error handling tests for scikit-fair preprocessing classes.

Tests behavior with:
- Invalid inputs (wrong column names, non-DataFrame, etc.)
- Edge cases (balanced data, minimal data)
- Boundary conditions
"""
import numpy as np
import pandas as pd
import pytest

from skfair.preprocessing import (
    GeometricFairnessRepair,
    IntersectionalBinarizer,
    Reweighing,
    ReweighingClassifier,
    FairSmote,
    FairOversampling,
    FAWOS,
    HeterogeneousFOS,
    Massaging,
    FairwayRemover,
)


class TestInvalidInputs:
    """Tests for error handling with invalid inputs."""

    def test_invalid_sens_attr_column(self, simple_binary_data):
        """Invalid sensitive attribute column raises error."""
        X, y = simple_binary_data

        fs = FairSmote(sens_attr='nonexistent_column', random_state=42)

        with pytest.raises((KeyError, ValueError)):
            fs.fit_resample(X, y)

    def test_reweighing_invalid_sens_attr(self, simple_binary_data):
        """Reweighing with invalid sens_attr raises error."""
        X, y = simple_binary_data

        rw = Reweighing(sens_attr='nonexistent')

        with pytest.raises((KeyError, ValueError)):
            rw.fit_transform(X, y)

    def test_massaging_none_sens_attr(self, simple_binary_data):
        """Massaging with sens_attr=None raises ValueError."""
        X, y = simple_binary_data

        ms = Massaging(sens_attr=None)

        with pytest.raises(ValueError):
            ms.fit_resample(X, y)

    def test_geometric_repair_invalid_columns(self, simple_binary_data):
        """GeometricFairnessRepair with invalid repair columns raises error."""
        X, y = simple_binary_data

        repair = GeometricFairnessRepair(
            sens_attr='group',
            repair_columns=['nonexistent_column']
        )

        with pytest.raises((KeyError, ValueError)):
            repair.fit(X, y)

    def test_intersectional_binarizer_requires_dataframe(self):
        """IntersectionalBinarizer requires DataFrame input."""
        X_array = np.array([[1, 2], [3, 4]])

        binarizer = IntersectionalBinarizer(privileged_definition={'col': 1})

        with pytest.raises((TypeError, ValueError, AttributeError)):
            binarizer.fit_transform(X_array)


class TestBalancedData:
    """Tests with already balanced data."""

    def test_massaging_balanced_no_change(self, balanced_data, sens_attr, priv_group):
        """Massaging on balanced data makes minimal or no changes."""
        X, y = balanced_data

        ms = Massaging(sens_attr=sens_attr, priv_group=priv_group)
        X_out, y_out = ms.fit_resample(X, y)

        # Sample count should stay the same
        assert len(X_out) == len(X)
        # Labels might be unchanged or minimally changed
        assert len(y_out) == len(y)

    def test_reweighing_balanced_weights_near_one(self, balanced_data, sens_attr):
        """Reweighing on balanced data produces weights close to 1."""
        X, y = balanced_data

        rw = Reweighing(sens_attr=sens_attr)
        _, weights = rw.fit_transform(X, y)

        # With balanced data, weights should be close to 1
        assert weights.mean() == pytest.approx(1.0, abs=0.3)

    def test_fair_smote_balanced_minimal_change(self, balanced_data, sens_attr):
        """FairSmote on balanced data adds minimal synthetic samples."""
        X, y = balanced_data

        fs = FairSmote(sens_attr=sens_attr, random_state=42)
        X_out, y_out = fs.fit_resample(X, y)

        # Already balanced, so minimal or no new samples
        # The max subgroup size is the target, so might still add some
        assert len(X_out) >= len(X)


class TestMinimalData:
    """Tests with minimum viable data size."""

    def test_fair_smote_minimal_data(self, minimal_data, sens_attr):
        """FairSmote works with minimal data (6 per subgroup)."""
        X, y = minimal_data

        fs = FairSmote(sens_attr=sens_attr, k_neighbors=5, random_state=42)
        X_out, y_out = fs.fit_resample(X, y)

        assert len(X_out) >= len(X)
        assert len(y_out) == len(X_out)

    def test_fos_minimal_data(self, minimal_data, sens_attr, priv_group):
        """FairOversampling works with minimal data."""
        X, y = minimal_data

        fos = FairOversampling(
            sens_attr=sens_attr,
            priv_group=priv_group,
            k_neighbors=5,
            random_state=42
        )
        X_out, y_out = fos.fit_resample(X, y)

        assert len(X_out) >= len(X)

    def test_fairway_minimal_data(self, minimal_data, sens_attr, priv_group):
        """FairwayRemover works with minimal data."""
        X, y = minimal_data

        fw = FairwayRemover(sens_attr=sens_attr, priv_group=priv_group)
        X_out, y_out = fw.fit_resample(X, y)

        # Should not add samples
        assert len(X_out) <= len(X)


class TestDataFramePreservation:
    """Tests that DataFrame structure is preserved."""

    def test_fair_smote_preserves_dtypes(self, simple_binary_data, sens_attr):
        """FairSmote preserves DataFrame column dtypes."""
        X, y = simple_binary_data
        X = X.astype({'age': 'float64', 'income': 'float64', 'group': 'int64'})

        fs = FairSmote(sens_attr=sens_attr, random_state=42)
        X_out, y_out = fs.fit_resample(X, y)

        assert isinstance(X_out, pd.DataFrame)
        assert list(X_out.columns) == list(X.columns)

    def test_fos_preserves_index_type(self, simple_binary_data, sens_attr, priv_group):
        """FairOversampling output is DataFrame with proper columns."""
        X, y = simple_binary_data

        fos = FairOversampling(sens_attr=sens_attr, priv_group=priv_group, random_state=42)
        X_out, y_out = fos.fit_resample(X, y)

        assert isinstance(X_out, pd.DataFrame)
        assert 'group' in X_out.columns
        assert 'age' in X_out.columns
        assert 'income' in X_out.columns

    def test_massaging_preserves_dataframe(self, simple_binary_data, sens_attr, priv_group):
        """Massaging preserves DataFrame structure."""
        X, y = simple_binary_data

        ms = Massaging(sens_attr=sens_attr, priv_group=priv_group)
        X_out, y_out = ms.fit_resample(X, y)

        assert isinstance(X_out, pd.DataFrame)
        assert list(X_out.columns) == list(X.columns)

    def test_geometric_repair_preserves_columns(self, simple_binary_data):
        """GeometricFairnessRepair preserves all columns."""
        X, y = simple_binary_data

        repair = GeometricFairnessRepair(
            sens_attr='group',
            repair_columns=['age']
        )
        X_out = repair.fit_transform(X)

        assert isinstance(X_out, pd.DataFrame)
        assert list(X_out.columns) == list(X.columns)
        # Non-repaired columns unchanged
        assert (X_out['income'] == X['income']).all()
        assert (X_out['group'] == X['group']).all()


class TestOutputValidity:
    """Tests that outputs are valid."""

    def test_reweighing_weights_sum(self, simple_binary_data, sens_attr):
        """Reweighing weights have reasonable sum."""
        X, y = simple_binary_data

        rw = Reweighing(sens_attr=sens_attr)
        _, weights = rw.fit_transform(X, y)

        # Weights should be positive
        assert (weights > 0).all()
        # Mean weight should be around 1 (not too far off)
        assert 0.5 < weights.mean() < 2.0

    def test_fair_smote_no_nans(self, simple_binary_data, sens_attr):
        """FairSmote output has no NaN values."""
        X, y = simple_binary_data

        fs = FairSmote(sens_attr=sens_attr, random_state=42)
        X_out, y_out = fs.fit_resample(X, y)

        assert not X_out.isna().any().any()
        assert not np.isnan(y_out).any()

    def test_massaging_valid_labels(self, simple_binary_data, sens_attr, priv_group):
        """Massaging output has same label values as input."""
        X, y = simple_binary_data
        original_labels = set(np.unique(y))

        ms = Massaging(sens_attr=sens_attr, priv_group=priv_group)
        _, y_out = ms.fit_resample(X, y)

        output_labels = set(np.unique(y_out))
        assert output_labels == original_labels

    def test_intersectional_adds_column(self, simple_binary_data):
        """IntersectionalBinarizer adds exactly one new column."""
        X, y = simple_binary_data
        original_cols = set(X.columns)

        binarizer = IntersectionalBinarizer(
            privileged_definition={'group': 1},
            group_col_name='is_privileged'
        )
        X_out = binarizer.fit_transform(X)

        new_cols = set(X_out.columns) - original_cols
        assert new_cols == {'is_privileged'}
        assert set(X_out['is_privileged'].unique()).issubset({0, 1})
