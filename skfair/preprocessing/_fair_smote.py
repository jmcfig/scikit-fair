"""
Fair-SMOTE oversampler for fairness-aware data balancing.

Balances the dataset across all (class_label × sensitive_attribute)
subgroups using a Differential Evolution-style crossover.
"""

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from ._base import BaseFairSampler


class FairSmote(BaseFairSampler):
    """
    Fair-SMOTE oversampler for fairness-aware data balancing.

    Balances the dataset across all (class_label × sensitive_attribute)
    subgroups. Identifies the largest subgroup and oversamples all others
    to match that size using a Differential Evolution-style crossover.

    Parameters
    ----------
    sens_attr : str
        Name of the sensitive attribute column in X.

    cr : float, default=0.8
        Crossover rate [0, 1]. Probability that a synthetic feature value
        differs from the parent.

    f : float, default=0.8
        Mutation amount [0, 1]. Magnitude of the differential step for
        numeric features: new = parent + f * (neighbor1 - neighbor2).

    k_neighbors : int, default=5
        Number of neighbors to use for KNN.

    clip_numeric : bool, default=True
        Whether to clip synthetic numeric values to the min/max range
        observed in the original data.

    random_state : int, RandomState instance or None, default=None
        Control the randomization of the algorithm.

    Attributes
    ----------
    subgroup_counts_ : dict
        The counts of samples per (class, sensitive_attr) subgroup.

    max_size_ : int
        The target size (largest subgroup count).
    """

    _sampling_type = "over-sampling"

    def __init__(
        self,
        sens_attr,
        cr=0.8,
        f=0.8,
        k_neighbors=5,
        clip_numeric=True,
        random_state=None
    ):
        super().__init__(sens_attr, random_state)
        self.cr = cr
        self.f = f
        self.k_neighbors = k_neighbors
        self.clip_numeric = clip_numeric

    def _generate_sample(self, parent, c1, c2, schema, rng):
        """
        Generate a single synthetic sample using Differential Evolution logic.
        """
        synth_row = {}
        numeric_cols = schema['numeric_cols']
        dtypes = schema['dtypes']
        bounds = schema.get('bounds', {})

        for col in parent.index:
            # Skip sensitive attribute (fixed for subgroup)
            if col == self.sens_attr:
                synth_row[col] = parent[col]
                continue

            # Skip if crossover check fails (keep parent value)
            if rng.rand() >= self.cr:
                synth_row[col] = parent[col]
                continue

            # Numeric Mutation
            if col in numeric_cols:
                new_val = self._interpolate_de(
                    float(parent[col]),
                    float(c1[col]),
                    float(c2[col]),
                    self.f
                )

                # Clip bounds
                if self.clip_numeric and col in bounds:
                    low, high = bounds[col]
                    new_val = np.clip(new_val, low, high)

                # Round if originally integer
                if pd.api.types.is_integer_dtype(dtypes[col]):
                    new_val = round(new_val)

                synth_row[col] = new_val

            # Categorical/Boolean Mutation (Random Selection)
            else:
                synth_row[col] = rng.choice([parent[col], c1[col], c2[col]])

        return synth_row

    def _fit_resample(self, X, y):  # noqa: ARG002
        """Required by BaseSampler but not used - we override fit_resample directly."""
        raise NotImplementedError("Use fit_resample() directly")

    def fit_resample(self, X, y):
        """
        Resample the dataset to balance all (class, sensitive_attr) subgroups.

        Overrides BaseSampler.fit_resample to preserve DataFrame input.

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix.
        y : array-like
            Target labels.

        Returns
        -------
        X_resampled : pandas DataFrame
            Resampled feature matrix.
        y_resampled : ndarray
            Resampled target labels.
        """
        # Validate input (uses base class method indirectly)
        self._check_X_y(X, y)
        rng = check_random_state(self.random_state)

        # Extract schema using base class method
        schema = self._get_schema(
            X,
            exclude_cols=[self.sens_attr],
            compute_bounds=self.clip_numeric
        )
        numeric_cols = schema['numeric_cols']
        knn_cols = numeric_cols

        # Standardize y as array for grouping
        y = np.array(y)

        # Get subgroups using base class method
        grouped_indices, self.subgroup_counts_ = self._get_subgroup_indices(X, y)

        target_size = max(self.subgroup_counts_.values())
        self.max_size_ = target_size

        # Lists to hold the final result
        resampled_rows = []
        resampled_y = []

        # Iterate Subgroups and Oversample
        for (label, p_val), indices in grouped_indices.items():
            count = len(indices)

            # Add existing data first
            subgroup_data = X.iloc[indices]
            resampled_rows.append(subgroup_data)
            resampled_y.extend([label] * count)

            n_needed = target_size - count
            if n_needed <= 0:
                continue

            # Prepare for KNN
            if knn_cols:
                X_knn = subgroup_data[knn_cols].values
            else:
                X_knn = pd.get_dummies(subgroup_data).values

            n_samples = len(subgroup_data)

            # Handle tiny subgroups (fallback to random duplication if < 3 samples)
            if n_samples < 3:
                dup_rows = subgroup_data.sample(n=n_needed, replace=True, random_state=rng)
                resampled_rows.append(dup_rows)
                resampled_y.extend([label] * n_needed)
                continue

            # Fit KNN using base class method
            nn, _ = self._fit_knn(X_knn, self.k_neighbors)

            if nn is None:
                # Fallback
                dup_rows = subgroup_data.sample(n=n_needed, replace=True, random_state=rng)
                resampled_rows.append(dup_rows)
                resampled_y.extend([label] * n_needed)
                continue

            # Generate Synthetic Samples
            new_samples = []

            for _ in range(n_needed):
                # Pick random parent
                parent_idx = rng.randint(n_samples)
                parent_row = subgroup_data.iloc[parent_idx]

                # Get 2 neighbors for DE mutation using base class method
                neighbor_indices = self._query_neighbors(nn, X_knn, parent_idx, n_select=2, rng=rng)

                c1_row = subgroup_data.iloc[neighbor_indices[0]]
                c2_row = subgroup_data.iloc[neighbor_indices[1]]

                # Generate
                synth_row = self._generate_sample(parent_row, c1_row, c2_row, schema, rng)
                new_samples.append(synth_row)

            # Add synthetic batch
            if new_samples:
                resampled_rows.append(pd.DataFrame(new_samples))
                resampled_y.extend([label] * n_needed)

        # Reconstruct Final DataFrame
        X_resampled = pd.concat(resampled_rows, ignore_index=True)
        y_resampled = np.array(resampled_y)

        # Final Type Enforcement using base class method
        self._enforce_schema(X_resampled, schema)

        return X_resampled, y_resampled
