"""
Fair Oversampling using Heterogeneous Clusters - Sonoda et al. (2023)

Oversamples all (class, group) subgroups to match the maximum subgroup size
using interpolation with heterogeneous clusters (different class OR different group).
"""

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors

from ._base import BaseFairSampler


class HeterogeneousFOS(BaseFairSampler):
    """
    Fair Oversampling using Heterogeneous Clusters by Sonoda et al. (2023).

    Balances all (class_label x sensitive_group) subgroups to match the largest
    subgroup size. Uses heterogeneous clusters for interpolation:
    - H_y: samples with different class but same group (class-heterogeneous)
    - H_g: samples with same class but different group (group-heterogeneous)

    A Bernoulli probability p_{y,g} determines which cluster to use for each
    synthetic sample, enabling generation of class-mix and group-mix features
    that improve classifier generalization and fairness.

    Parameters
    ----------
    sens_attr : str
        Name of the sensitive attribute column in X.

    k_neighbors : int, default=5
        Number of nearest neighbors for computing local density and
        interpolation normalization.

    random_state : int, RandomState instance or None, default=None
        Controls randomization for reproducibility.

    Attributes
    ----------
    subgroup_counts_ : dict
        Sample counts per (class_label, sensitive_group) after fit.

    n_synthetic_ : dict
        Number of synthetic samples generated per subgroup.

    max_size_ : int
        The target size (largest subgroup count).

    References
    ----------
    Sonoda, R. (2023). Fair Oversampling Technique using Heterogeneous Clusters.
    arXiv:2305.13875v1
    """

    _sampling_type = "over-sampling"

    def __init__(self, sens_attr, k_neighbors=5, random_state=None):
        super().__init__(sens_attr, random_state)
        self.k_neighbors = k_neighbors

    def _compute_local_density(self, point_idx, nn, X_knn, y):
        """
        Compute local density Delta_i for a point (Eq. 9).

        Delta_i is the fraction of KNN that have different class than the point.
        Used to determine the interpolation weight distribution to avoid overlap.

        Parameters
        ----------
        point_idx : int
            Index of the point.
        nn : NearestNeighbors
            Fitted KNN model.
        X_knn : ndarray
            Feature matrix for KNN.
        y : ndarray
            Class labels.

        Returns
        -------
        delta : float
            Local density in [0, 1].
        """
        _, neighbors = nn.kneighbors(X_knn[point_idx].reshape(1, -1))
        neighbors = neighbors[0][1:]  # Exclude self

        if len(neighbors) == 0:
            return 1.0

        point_class = y[point_idx]
        different_class_count = sum(1 for n in neighbors if y[n] != point_class)

        return different_class_count / len(neighbors)

    def _get_max_knn_distance(self, point_idx, nn, X_knn):
        """
        Get the maximum distance to K nearest neighbors.

        Parameters
        ----------
        point_idx : int
            Index of the point.
        nn : NearestNeighbors
            Fitted KNN model.
        X_knn : ndarray
            Feature matrix for KNN.

        Returns
        -------
        max_dist : float
            Maximum distance to KNN.
        """
        distances, _ = nn.kneighbors(X_knn[point_idx].reshape(1, -1))
        distances = distances[0][1:]  # Exclude self (distance 0)

        if len(distances) == 0:
            return 1.0

        return np.max(distances)

    def _euclidean_distance(self, x1, x2):
        """Compute Euclidean distance between two vectors."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _fit_resample(self, X, y):
        """
        Resample using Fair Heterogeneous Oversampling algorithm.

        Algorithm (Sonoda et al. 2023):
        1. Create clusters C_{y,g} for each (class, group) combination
        2. Find maximum cluster size S
        3. For each cluster smaller than S:
           a. Sample x_i from cluster
           b. Compute p_{y,g} = |H_y| / (|H_y| + |H_g|)
           c. Draw b ~ Bernoulli(p_{y,g})
           d. Sample x_j from H_y (if b=True) or H_g (if b=False)
           e. Generate synthetic sample using Eq. (8)
        """
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        prot_values = X[self.sens_attr].values

        # Get unique classes and groups
        classes = np.unique(y)
        groups = np.unique(prot_values)

        if len(classes) != 2:
            raise ValueError(
                f"HeterogeneousFOS requires binary classification, got {len(classes)} classes."
            )

        # Extract schema for type preservation
        schema = self._get_schema(X, exclude_cols=[self.sens_attr])
        numeric_cols = schema['numeric_cols']

        if not numeric_cols:
            raise ValueError("HeterogeneousFOS requires at least one numeric feature.")

        # Fit global KNN on all data (for local density and distance normalization)
        X_knn = X[numeric_cols].values
        nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn.fit(X_knn)

        # Create clusters C_{y,g}
        clusters = {}
        for cls in classes:
            for grp in groups:
                mask = (y == cls) & (prot_values == grp)
                clusters[(cls, grp)] = np.where(mask)[0]

        # Record subgroup counts
        self.subgroup_counts_ = {k: len(v) for k, v in clusters.items()}

        # Find maximum cluster size S
        max_size = max(len(idx) for idx in clusters.values())
        self.max_size_ = max_size

        # Collect synthetic samples
        synthetic_rows = []
        synthetic_labels = []
        self.n_synthetic_ = {}

        for (cls, grp), indices in clusters.items():
            n_needed = max_size - len(indices)
            self.n_synthetic_[(cls, grp)] = max(0, n_needed)

            if n_needed <= 0 or len(indices) == 0:
                continue

            # Compose heterogeneous clusters
            # H_y: different class, same group (intra-group pairs -> class-mix features)
            H_y = np.where((y != cls) & (prot_values == grp))[0]
            # H_g: same class, different group (intra-class pairs -> group-mix features)
            H_g = np.where((y == cls) & (prot_values != grp))[0]

            # Compute probability p_{y,g} (Eq. 7)
            H_y_size = len(H_y)
            H_g_size = len(H_g)

            if H_y_size + H_g_size == 0:
                # No heterogeneous samples available, fallback to duplication
                if len(indices) > 0:
                    dup_idx = rng.choice(indices, size=n_needed, replace=True)
                    synthetic_rows.extend(X.iloc[dup_idx].to_dict('records'))
                    synthetic_labels.extend([cls] * n_needed)
                continue

            p_yg = H_y_size / (H_y_size + H_g_size)

            # Generate synthetic samples
            for _ in range(n_needed):
                # Randomly sample x_i from cluster C_{y,g}
                i = rng.choice(indices)
                x_i = X_knn[i]
                x_i_row = X.iloc[i]

                # Draw b ~ Bernoulli(p_{y,g})
                use_Hy = rng.random() < p_yg

                # Select x_j from heterogeneous cluster
                if use_Hy and H_y_size > 0:
                    j = rng.choice(H_y)
                elif H_g_size > 0:
                    j = rng.choice(H_g)
                elif H_y_size > 0:
                    j = rng.choice(H_y)
                else:
                    # Fallback: duplicate x_i
                    synthetic_rows.append(x_i_row.to_dict())
                    synthetic_labels.append(cls)
                    continue

                x_j = X_knn[j]
                x_j_row = X.iloc[j]

                # Compute local density Delta_i (Eq. 9)
                delta_i = self._compute_local_density(i, nn, X_knn, y)

                # If delta_i is 0 (no overlap), use small positive value
                if delta_i <= 0:
                    delta_i = 0.1

                # Draw interpolation weight w_i ~ U(0, Delta_i)
                w_i = rng.uniform(0, delta_i)

                # Compute distance normalization factor (Eq. 8)
                d_ij = self._euclidean_distance(x_i, x_j)
                max_knn_dist = self._get_max_knn_distance(i, nn, X_knn)

                if d_ij > 0:
                    norm_factor = max_knn_dist / d_ij
                else:
                    norm_factor = 1.0

                # Generate synthetic sample using Eq. (8)
                # x_new = x_i + w_i * (x_j - x_i) * (max_k d(i,KNN(i)) / d(i,j))
                synth_row = {}

                for col in X.columns:
                    if col == self.sens_attr:
                        # Preserve sensitive attribute from x_i (parent cluster)
                        synth_row[col] = x_i_row[col]
                    elif col in numeric_cols:
                        col_idx = numeric_cols.index(col)
                        diff = x_j[col_idx] - x_i[col_idx]
                        val = x_i[col_idx] + w_i * diff * norm_factor

                        if pd.api.types.is_integer_dtype(schema['dtypes'][col]):
                            val = round(val)
                        synth_row[col] = val
                    else:
                        # Categorical: random selection between x_i and x_j
                        synth_row[col] = rng.choice([x_i_row[col], x_j_row[col]])

                synthetic_rows.append(synth_row)
                synthetic_labels.append(cls)

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
