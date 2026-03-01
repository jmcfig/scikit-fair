"""
Disparate Impact Remover (Feldman et al., 2015)

Implements the Geometric Repair algorithm from:
"Certifying and Removing Disparate Impact" - Feldman, Friedler, Moeller,
Scheidegger, Venkatasubramanian (KDD 2015)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GeometricFairnessRepair(BaseEstimator, TransformerMixin):
    """
    Geometric Disparate Impact Remover (Feldman et al., 2015).

    Transforms feature distributions to reduce correlation with a sensitive
    attribute while preserving within-group rank ordering. Uses the quantile
    bucket method to handle uneven group sizes.

    The repair formula (Definition 5.2):
        ȳ = (1 - λ) * y + λ * F_A⁻¹(F_x(y))

    Where:
    - y is the original feature value
    - λ (lambda_param) controls repair level: 0 = no change, 1 = full repair
    - F_x(y) maps y to its quantile rank within group x
    - F_A⁻¹ maps that rank to the median distribution value

    Parameters
    ----------
    sens_attr : str
        Column name of the sensitive attribute in X.

    repair_columns : list of str
        Column names to apply repair to.

    lambda_param : float, default=1.0
        Repair level between 0.0 (no repair) and 1.0 (full repair).

    Attributes
    ----------
    n_buckets_ : int
        Number of quantile buckets (equals smallest group size).

    bucket_edges_ : dict
        For each (column, group): array of quantile edges.

    group_medians_ : dict
        For each (column, group): array of bucket medians.

    median_distribution_ : dict
        For each column: array of median values across all groups per bucket.

    groups_ : array
        Unique values of the sensitive attribute.

    Example
    -------
    >>> from skfair.preprocessing import GeometricFairnessRepair
    >>> repair = GeometricFairnessRepair(
    ...     sens_attr='sex',
    ...     repair_columns=['age', 'income'],
    ...     lambda_param=1.0
    ... )
    >>> X_repaired = repair.fit_transform(X)
    """

    def __init__(self, sens_attr, repair_columns, lambda_param=1.0):
        self.sens_attr = sens_attr
        self.repair_columns = repair_columns
        self.lambda_param = lambda_param

    def _validate_params(self, X):
        """Validate inputs before fitting."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if self.sens_attr not in X.columns:
            raise ValueError(
                f"Protected attribute '{self.sens_attr}' not in X."
            )

        missing_cols = set(self.repair_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Repair columns not found in X: {missing_cols}")

        if not 0.0 <= self.lambda_param <= 1.0:
            raise ValueError(
                f"lambda_param must be between 0 and 1, got {self.lambda_param}"
            )

    def fit(self, X, y=None):
        """
        Learn the quantile bucket structure and median distribution.

        For each repair column:
        1. Compute N_min (smallest group size) → number of buckets
        2. Divide each group into N_min quantile buckets
        3. Compute the median of each bucket per group
        4. Compute the median distribution A (median across groups)

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix containing sens_attr and repair_columns.

        y : ignored
            Not used, present for sklearn API compatibility.

        Returns
        -------
        self
        """
        self._validate_params(X)

        # Get sensitive attribute groups
        self.groups_ = X[self.sens_attr].unique()

        # Compute group sizes and N_min
        group_sizes = X.groupby(self.sens_attr).size()
        self.n_buckets_ = int(group_sizes.min())

        if self.n_buckets_ < 2:
            raise ValueError(
                f"Smallest group has only {self.n_buckets_} samples. "
                "Need at least 2 for meaningful quantile buckets."
            )

        # Quantile points for bucket edges (0, 1/n, 2/n, ..., 1)
        quantile_points = np.linspace(0, 1, self.n_buckets_ + 1)

        self.bucket_edges_ = {}
        self.group_medians_ = {}
        self.median_distribution_ = {}

        for col in self.repair_columns:
            col_group_medians = {}

            for group in self.groups_:
                # Get values for this group
                mask = X[self.sens_attr] == group
                values = X.loc[mask, col].values

                # Compute bucket edges (quantiles)
                edges = np.quantile(values, quantile_points)
                self.bucket_edges_[(col, group)] = edges

                # Compute median of each bucket
                # Assign each value to a bucket, then compute medians
                bucket_indices = np.digitize(values, edges[1:-1])  # n_buckets buckets

                bucket_medians = []
                for b in range(self.n_buckets_):
                    bucket_values = values[bucket_indices == b]
                    if len(bucket_values) > 0:
                        bucket_medians.append(np.median(bucket_values))
                    else:
                        # Fallback: use edge midpoint if bucket is empty
                        bucket_medians.append((edges[b] + edges[b + 1]) / 2)

                bucket_medians = np.array(bucket_medians)
                self.group_medians_[(col, group)] = bucket_medians
                col_group_medians[group] = bucket_medians

            # Compute median distribution A: median across groups for each bucket
            # Stack all group medians: shape (n_groups, n_buckets)
            all_medians = np.array([col_group_medians[g] for g in self.groups_])
            # Median across groups (axis=0) for each bucket
            self.median_distribution_[col] = np.median(all_medians, axis=0)

        return self

    def transform(self, X):
        """
        Apply geometric repair to the specified columns.

        For each value:
        1. Find which group the sample belongs to
        2. Find which bucket the value falls into (F_x mapping)
        3. Get the median distribution value for that bucket (F_A⁻¹)
        4. Apply: ȳ = (1 - λ) * y + λ * repaired_value

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix to transform.

        Returns
        -------
        X_repaired : pandas DataFrame
            Transformed feature matrix with repaired columns.
        """
        X_repaired = X.copy()

        # If lambda is 0, no repair needed
        if self.lambda_param == 0.0:
            return X_repaired

        for col in self.repair_columns:
            repaired_values = np.zeros(len(X))

            for group in self.groups_:
                mask = X[self.sens_attr] == group
                if not mask.any():
                    continue

                values = X.loc[mask, col].values
                edges = self.bucket_edges_[(col, group)]

                # Clip values to training range (handle out-of-range)
                values_clipped = np.clip(values, edges[0], edges[-1])

                # Find bucket indices (F_x mapping)
                # digitize with edges[1:-1] gives buckets 0 to n_buckets-1
                bucket_indices = np.digitize(values_clipped, edges[1:-1])
                # Clip to valid bucket range (0 to n_buckets-1)
                bucket_indices = np.clip(bucket_indices, 0, self.n_buckets_ - 1)

                # Get repaired values from median distribution (F_A⁻¹)
                median_dist = self.median_distribution_[col]
                repaired = median_dist[bucket_indices]

                # Apply geometric repair formula
                # ȳ = (1 - λ) * y + λ * repaired
                final_values = (1 - self.lambda_param) * values + self.lambda_param * repaired

                repaired_values[mask] = final_values

            X_repaired[col] = repaired_values

        return X_repaired
