"""
Fair Oversampling (FOS) - Dablan et al.

Oversamples minority class instances within each sensitive group
to balance class distributions while maintaining fairness.
"""

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from ._base import BaseFairSampler


class FairOversampling(BaseFairSampler):
    """
    Fair Oversampling (FOS) by Dablan et al.

    Balances class distributions within each sensitive group independently.
    For each group (privileged/unprivileged), oversamples the minority class
    to match the majority class count using SMOTE-style interpolation.

    Parameters
    ----------
    sens_attr : str
        Name of the sensitive attribute column in X.

    priv_group : int or str
        Value in `sens_attr` that represents the privileged group.

    k_neighbors : int, default=5
        Number of nearest neighbors for SMOTE interpolation.

    random_state : int, RandomState instance or None, default=None
        Controls randomization for reproducibility.

    Attributes
    ----------
    subgroup_counts_ : dict
        Sample counts per (sensitive_group, class_label) after fit.

    n_synthetic_ : dict
        Number of synthetic samples generated per group.
    """

    _sampling_type = "over-sampling"

    def __init__(self, sens_attr, priv_group, k_neighbors=5, random_state=None):
        super().__init__(sens_attr, random_state)
        self.priv_group = priv_group
        self.k_neighbors = k_neighbors

    def _fit_resample(self, X, y):
        """
        Resample the dataset using Fair Oversampling algorithm.

        Algorithm (Dablan et al.):
        1. Split data into 4 subgroups: (priv/unpriv) × (maj/min class)
        2. Calculate samples needed: S_pr = N_pr_maj - N_pr_min, S_up = N_up_maj - N_up_min
        3. Oversample smaller deficit first, then larger
        4. Use SMOTE interpolation: S = B + R × (N - B)
        """
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError(f"FOS requires binary classification, got {len(classes)} classes.")

        # Identify majority/minority class (globally)
        class_counts = {c: np.sum(y == c) for c in classes}
        maj_class = max(class_counts, key=class_counts.get)
        min_class = min(class_counts, key=class_counts.get)

        # Split into 4 subgroups
        mask_priv = (X[self.sens_attr] == self.priv_group).values
        mask_unpriv = ~mask_priv

        idx_pr_maj = np.where(mask_priv & (y == maj_class))[0]
        idx_pr_min = np.where(mask_priv & (y == min_class))[0]
        idx_up_maj = np.where(mask_unpriv & (y == maj_class))[0]
        idx_up_min = np.where(mask_unpriv & (y == min_class))[0]

        self.subgroup_counts_ = {
            ('priv', 'maj'): len(idx_pr_maj),
            ('priv', 'min'): len(idx_pr_min),
            ('unpriv', 'maj'): len(idx_up_maj),
            ('unpriv', 'min'): len(idx_up_min),
        }

        # Calculate oversampling amounts
        S_pr = len(idx_pr_maj) - len(idx_pr_min)
        S_up = len(idx_up_maj) - len(idx_up_min)

        # Determine order: oversample smaller deficit first
        if S_up < S_pr:
            tasks = [
                (S_up, idx_up_min, 'unpriv'),
                (S_pr, idx_pr_min, 'priv'),
            ]
        else:
            tasks = [
                (S_pr, idx_pr_min, 'priv'),
                (S_up, idx_up_min, 'unpriv'),
            ]

        # Extract schema using base class method
        schema = self._get_schema(X, exclude_cols=[self.sens_attr])
        numeric_cols = schema['numeric_cols']

        if not numeric_cols:
            raise ValueError("FOS requires at least one numeric feature for KNN.")

        # Collect synthetic samples
        synthetic_rows = []
        synthetic_labels = []
        self.n_synthetic_ = {}

        for n_samples, minority_idx, group_name in tasks:
            if n_samples <= 0 or len(minority_idx) == 0:
                self.n_synthetic_[group_name] = 0
                continue

            X_minority = X.iloc[minority_idx]
            X_knn = X_minority[numeric_cols].values

            # Fit KNN using base class method
            nn, _ = self._fit_knn(X_knn, self.k_neighbors)

            if nn is None:
                # Too few samples for KNN, duplicate randomly
                dup_idx = rng.choice(minority_idx, size=n_samples, replace=True)
                synthetic_rows.extend(X.iloc[dup_idx].to_dict('records'))
                synthetic_labels.extend([min_class] * n_samples)
                self.n_synthetic_[group_name] = n_samples
                continue

            # Generate synthetic samples
            base_local_idx = rng.choice(len(minority_idx), size=n_samples, replace=True)

            for local_idx in base_local_idx:
                base_row = X_minority.iloc[local_idx]

                # Get one neighbor using base class method
                neighbor_local = self._query_neighbors(nn, X_knn, local_idx, n_select=1, rng=rng)
                neighbor_row = X_minority.iloc[neighbor_local[0]]

                # Generate synthetic sample using base class interpolation
                r = rng.random()
                synth_row = {}

                for col in X.columns:
                    if col == self.sens_attr:
                        synth_row[col] = base_row[col]
                    elif col in numeric_cols:
                        val = self._interpolate_smote(float(base_row[col]), float(neighbor_row[col]), r)
                        if pd.api.types.is_integer_dtype(schema['dtypes'][col]):
                            val = round(val)
                        synth_row[col] = val
                    else:
                        synth_row[col] = rng.choice([base_row[col], neighbor_row[col]])

                synthetic_rows.append(synth_row)
                synthetic_labels.append(min_class)

            self.n_synthetic_[group_name] = n_samples

        # Combine original and synthetic
        if synthetic_rows:
            synth_df = pd.DataFrame(synthetic_rows)
            self._enforce_schema(synth_df, schema)
            X_resampled = pd.concat([X, synth_df], ignore_index=True)
            y_resampled = np.concatenate([y, synthetic_labels])
        else:
            X_resampled = X.copy()
            y_resampled = y.copy()

        return X_resampled, y_resampled
