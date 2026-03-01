"""
Base class for fairness-aware samplers.

Provides common functionality for all fair resampling algorithms:
- DataFrame preservation (prevents imblearn from converting to numpy)
- Protected attribute validation
- Schema extraction and type enforcement
- KNN operations for synthetic sample generation
- Interpolation methods (SMOTE, DE)
"""

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler

from ._sampler_utils import (
    validate_sampler_input,
    extract_numeric_schema,
    fit_knn,
    query_neighbors,
    enforce_dtypes,
    interpolate_smote,
    interpolate_de,
)


class BaseFairSampler(BaseSampler):
    """
    Base class for fairness-aware samplers.

    Extends imblearn's BaseSampler with common functionality for fair resampling
    algorithms. Ensures DataFrame inputs are preserved and provides utilities for
    sensitive attribute handling, KNN operations, and synthetic sample generation.

    Parameters
    ----------
    sens_attr : str
        Name of the sensitive attribute column in X (DataFrame).

    random_state : int, RandomState instance or None, default=None
        Controls randomization for reproducibility.

    Notes
    -----
    Subclasses must:
    - Set `_sampling_type` class attribute ("over-sampling" or "clean-sampling")
    - Implement `_fit_resample(X, y)` method
    - Call `super().__init__(sens_attr, random_state)` in `__init__`
    """

    # Subclasses must override this
    _sampling_type = "over-sampling"

    def __init__(self, sens_attr, random_state=None):
        super().__init__()
        self.sens_attr = sens_attr
        self.random_state = random_state

    # =========================================================================
    # DataFrame Preservation (Critical Override)
    # =========================================================================

    def _check_X_y(self, X, y):
        """
        Override imblearn's _check_X_y to preserve DataFrame structure.

        Returns False for the binarize_y flag to prevent DataFrame conversion
        to numpy arrays.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : array-like
            Target labels.

        Returns
        -------
        X : pd.DataFrame
            Unchanged feature matrix.
        y : array-like
            Unchanged target labels.
        bool
            False to prevent binarization.
        """
        validate_sampler_input(X, self.sens_attr)
        return X, y, False

    # =========================================================================
    # Schema and Type Preservation Methods
    # =========================================================================

    def _get_schema(self, X, exclude_cols=None, compute_bounds=False):
        """
        Extract schema information for type preservation.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        exclude_cols : list of str, optional
            Columns to exclude from numeric detection (e.g., sens_attr).
        compute_bounds : bool, default=False
            Whether to compute min/max bounds for numeric columns.

        Returns
        -------
        dict
            Schema with keys: 'dtypes', 'numeric_cols', 'bounds' (optional).
        """
        if exclude_cols is None:
            exclude_cols = []

        return extract_numeric_schema(
            X,
            exclude_cols=exclude_cols,
            compute_bounds=compute_bounds
        )

    def _enforce_schema(self, df, schema):
        """
        Enforce original DataFrame dtypes on resampled data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to restore dtypes on (modified in place).
        schema : dict
            Schema dictionary from _get_schema().
        """
        enforce_dtypes(df, schema['dtypes'])

    # =========================================================================
    # KNN and Neighbor Methods
    # =========================================================================

    def _fit_knn(self, X_numeric, k_neighbors):
        """
        Fit a KNN model with edge case handling.

        Parameters
        ----------
        X_numeric : ndarray of shape (n_samples, n_features)
            Numeric feature matrix.
        k_neighbors : int
            Desired number of neighbors.

        Returns
        -------
        nn : NearestNeighbors or None
            Fitted KNN model, or None if insufficient samples.
        k : int
            Adjusted k value.
        """
        return fit_knn(X_numeric, k_neighbors)

    def _query_neighbors(self, nn, X_numeric, idx, n_select=1, rng=None):
        """
        Query neighbors for a sample, excluding self.

        Parameters
        ----------
        nn : NearestNeighbors
            Fitted KNN model.
        X_numeric : ndarray
            Feature matrix used to fit KNN.
        idx : int
            Index of the query sample.
        n_select : int, default=1
            Number of neighbors to randomly select.
        rng : RandomState or None
            Random state for selection.

        Returns
        -------
        ndarray
            Array of selected neighbor indices.
        """
        return query_neighbors(nn, X_numeric, idx, n_select=n_select, rng=rng)

    # =========================================================================
    # Synthetic Sample Generation Methods
    # =========================================================================

    def _interpolate_smote(self, parent_val, neighbor_val, r):
        """
        SMOTE-style linear interpolation.

        Formula: S = parent + r * (neighbor - parent)

        Parameters
        ----------
        parent_val : float
            Parent/base value.
        neighbor_val : float
            Neighbor value.
        r : float
            Random value in [0, 1].

        Returns
        -------
        float
            Interpolated value.
        """
        return interpolate_smote(parent_val, neighbor_val, r)

    def _interpolate_de(self, parent_val, n1_val, n2_val, f):
        """
        Differential Evolution-style mutation.

        Formula: S = parent + f * (n1 - n2)

        Parameters
        ----------
        parent_val : float
            Parent value.
        n1_val : float
            First neighbor value.
        n2_val : float
            Second neighbor value.
        f : float
            Mutation factor.

        Returns
        -------
        float
            Mutated value.
        """
        return interpolate_de(parent_val, n1_val, n2_val, f)

    # =========================================================================
    # Subgroup Analysis Methods
    # =========================================================================

    def _get_subgroup_indices(self, X, y, classes=None, prot_values=None):
        """
        Partition data into (class_label × sensitive_attribute) subgroups.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : array-like
            Target labels.
        classes : array-like, optional
            Class labels (inferred if not provided).
        prot_values : array-like, optional
            Protected attribute values (inferred if not provided).

        Returns
        -------
        subgroup_indices : dict
            Mapping {(class_label, prot_value): indices}.
        subgroup_counts : dict
            Mapping {(class_label, prot_value): count}.
        """
        y = np.asarray(y)
        if classes is None:
            classes = np.unique(y)
        if prot_values is None:
            prot_values = X[self.sens_attr].unique()

        subgroup_indices = {}
        subgroup_counts = {}

        for label in classes:
            for p_val in prot_values:
                mask = (y == label) & (X[self.sens_attr] == p_val)
                indices = np.where(mask)[0]
                subgroup_indices[(label, p_val)] = indices
                subgroup_counts[(label, p_val)] = len(indices)

        return subgroup_indices, subgroup_counts

    # =========================================================================
    # Abstract Method
    # =========================================================================

    def _fit_resample(self, X, y):
        """
        Resample the dataset (must be implemented by subclass).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : array-like
            Target labels.

        Returns
        -------
        X_resampled : pd.DataFrame
            Resampled feature matrix.
        y_resampled : array-like
            Resampled target labels.
        """
        raise NotImplementedError(
            "Subclasses must implement _fit_resample(X, y)"
        )
