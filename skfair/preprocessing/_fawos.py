"""
FAWOS: Fairness-Aware Oversampling - Salazar et al.

Oversamples positive unprivileged groups using typology-based weighted
selection to match the privileged/unprivileged positive-to-negative ratios.
"""

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors

from ._base import BaseFairSampler


class FAWOS(BaseFairSampler):
    """
    FAWOS: Fairness-Aware Oversampling by Salazar et al.

    Balances the dataset by oversampling positive unprivileged instances
    to match the ratio of positive-to-negative in the privileged group.
    Uses typology-based weighted selection (Safe, Borderline, Rare, Outlier)
    to prioritize points that are harder to learn.

    Parameters
    ----------
    sens_attr : str
        Name of the sensitive attribute column in X.

    priv_group : scalar
        Value in `sens_attr` that represents the privileged group.

    alpha : float, default=1.0
        Oversampling factor. When alpha=1, the positive-to-negative ratio
        of unprivileged groups will match the privileged group exactly.
        alpha < 1 creates fewer synthetic samples, alpha > 1 creates more.

    safe_weight : float, default=1.0
        Selection weight for Safe points (4-5 same-type neighbors).

    borderline_weight : float, default=1.0
        Selection weight for Borderline points (2-3 same-type neighbors).

    rare_weight : float, default=1.0
        Selection weight for Rare points (1 isolated same-type neighbor).

    random_state : int, RandomState instance or None, default=None
        Controls randomization for reproducibility.

    Attributes
    ----------
    n_synthetic_ : int
        Number of synthetic samples generated.

    typology_counts_ : dict
        Counts of each typology type in PU (positive unprivileged).

    Notes
    -----
    - k=5 is fixed per Napierala & Stefanowski (2016) typology thresholds.
    - SMOTE generation uses same-type neighbors (same Y and S) only,
      matching the authors' implementation, not global neighbors.

    References
    ----------
    Salazar, T., Santos, M. S., Araújo, H., & Abreu, P. H. (2021).
    FAWOS: Fairness-Aware Oversampling Algorithm Based on Distributions
    of Sensitive Attributes. IEEE Access, 9, 81370-81379.
    """

    _sampling_type = "over-sampling"
    _K_NEIGHBORS = 5  # Fixed per Napierala & Stefanowski (2016)

    def __init__(
        self,
        sens_attr,
        priv_group,
        alpha=1.0,
        safe_weight=1.0,
        borderline_weight=1.0,
        rare_weight=1.0,
        random_state=None
    ):
        super().__init__(sens_attr, random_state)
        self.priv_group = priv_group
        self.alpha = alpha
        self.safe_weight = safe_weight
        self.borderline_weight = borderline_weight
        self.rare_weight = rare_weight

    def _count_same_type_neighbors(self, point_idx, neighbor_indices, y, prot_values):
        """
        Count neighbors that have the same Y and same S as the point.

        Parameters
        ----------
        point_idx : int
            Global index of the point in the dataset.
        neighbor_indices : ndarray
            Indices of the k nearest neighbors (excluding self).
        y : ndarray
            Target labels for all samples.
        prot_values : ndarray
            Protected attribute values for all samples.

        Returns
        -------
        count : int
            Number of same-type neighbors.
        same_type_indices : list
            Indices of same-type neighbors.
        """
        point_y = y[point_idx]
        point_s = prot_values[point_idx]

        same_type_indices = []
        for n_idx in neighbor_indices:
            if y[n_idx] == point_y and prot_values[n_idx] == point_s:
                same_type_indices.append(n_idx)

        return len(same_type_indices), same_type_indices

    def _get_typology_weight(self, point_idx, nn, X_knn, y, prot_values):
        """
        Determine the typology weight for a point based on its neighborhood.

        Typology classification (Napierala & Stefanowski, 2016):
        - Safe (4-5 same-type neighbors): weight = safe_weight
        - Borderline (2-3 same-type neighbors): weight = borderline_weight
        - Rare (1 same-type neighbor, neighbor is isolated): weight = rare_weight
        - Outlier (0 same-type neighbors): weight = 0

        Parameters
        ----------
        point_idx : int
            Global index of the point.
        nn : NearestNeighbors
            Fitted KNN model on all data.
        X_knn : ndarray
            Numeric feature matrix used for KNN.
        y : ndarray
            Target labels.
        prot_values : ndarray
            Protected attribute values.

        Returns
        -------
        weight : float
            Typology-based selection weight.
        typology : str
            Typology label ('safe', 'borderline', 'rare', 'outlier').
        same_type_indices : list
            Indices of same-type neighbors (for use in generation).
        """
        # Get k neighbors (excluding self)
        _, all_neighbors = nn.kneighbors(X_knn[point_idx].reshape(1, -1))
        neighbors = all_neighbors[0][1:]  # Exclude self

        count, same_type_neighbors = self._count_same_type_neighbors(
            point_idx, neighbors, y, prot_values
        )

        if count >= 4:
            return self.safe_weight, 'safe', same_type_neighbors
        elif count >= 2:
            return self.borderline_weight, 'borderline', same_type_neighbors
        elif count == 1:
            # Check if the single neighbor is also isolated (Rare check)
            neighbor_idx = same_type_neighbors[0]
            _, neighbor_neighbors = nn.kneighbors(X_knn[neighbor_idx].reshape(1, -1))
            neighbor_neighbors = neighbor_neighbors[0][1:]

            neighbor_count, neighbor_same_type = self._count_same_type_neighbors(
                neighbor_idx, neighbor_neighbors, y, prot_values
            )

            # Rare: neighbor has 0 same-type, or 1 same-type which is the original point
            if neighbor_count == 0:
                return self.rare_weight, 'rare', same_type_neighbors
            elif neighbor_count == 1 and point_idx in neighbor_same_type:
                return self.rare_weight, 'rare', same_type_neighbors
            else:
                return self.borderline_weight, 'borderline', same_type_neighbors
        else:
            # Outlier: 0 same-type neighbors
            return 0.0, 'outlier', []

    def _fit_resample(self, X, y):
        """
        Resample the dataset using FAWOS algorithm.

        Algorithm:
        1. Compute PP, NP (privileged) and PU, NU (unprivileged) counts
        2. Calculate N = alpha * (|PP| * |NU| / |NP| - |PU|)
        3. Assign typology weights to each PU point
        4. Generate N synthetic samples via weighted selection + SMOTE
        """
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        prot_values = X[self.sens_attr].values

        # Identify privileged vs unprivileged
        mask_priv = prot_values == self.priv_group
        mask_unpriv = ~mask_priv

        # Assume binary classification with positive class = 1
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"FAWOS requires binary classification, got {len(classes)} classes.")

        # Determine positive class (assume 1, or the larger value)
        pos_class = max(classes)

        # Compute subgroup counts
        idx_pp = np.where(mask_priv & (y == pos_class))[0]  # Positive Privileged
        idx_np = np.where(mask_priv & (y != pos_class))[0]  # Negative Privileged
        idx_pu = np.where(mask_unpriv & (y == pos_class))[0]  # Positive Unprivileged
        idx_nu = np.where(mask_unpriv & (y != pos_class))[0]  # Negative Unprivileged

        n_pp = len(idx_pp)
        n_np = len(idx_np)
        n_pu = len(idx_pu)
        n_nu = len(idx_nu)

        # Calculate number of synthetic samples needed
        # N = alpha * (|PP| * |NU| / |NP| - |PU|)
        if n_np == 0:
            # Edge case: no negative privileged samples
            n_synthetic = 0
        else:
            n_synthetic = int(self.alpha * (n_pp * n_nu / n_np - n_pu))

        self.n_synthetic_ = max(0, n_synthetic)

        if self.n_synthetic_ <= 0 or n_pu == 0:
            # No oversampling needed or no PU samples to oversample from
            self.typology_counts_ = {'safe': 0, 'borderline': 0, 'rare': 0, 'outlier': 0}
            return X.copy(), y.copy()

        # Extract schema for type preservation
        schema = self._get_schema(X, exclude_cols=[self.sens_attr])
        numeric_cols = schema['numeric_cols']

        if not numeric_cols:
            raise ValueError("FAWOS requires at least one numeric feature for KNN.")

        # Fit global KNN on all data
        X_knn = X[numeric_cols].values
        nn = NearestNeighbors(n_neighbors=self._K_NEIGHBORS + 1)
        nn.fit(X_knn)

        # Compute typology weights and same-type neighbors for all PU points
        weights = []
        typologies = []
        same_type_neighbors_list = []  # Store same-type neighbors for generation
        self.typology_counts_ = {'safe': 0, 'borderline': 0, 'rare': 0, 'outlier': 0}

        for pu_idx in idx_pu:
            weight, typology, same_type_neighbors = self._get_typology_weight(
                pu_idx, nn, X_knn, y, prot_values
            )
            weights.append(weight)
            typologies.append(typology)
            same_type_neighbors_list.append(same_type_neighbors)
            self.typology_counts_[typology] += 1

        weights = np.array(weights)

        # Handle case where all weights are 0 (all outliers)
        if weights.sum() == 0:
            # Fall back to uniform weights for non-outliers, or duplicate if all outliers
            weights = np.ones(len(idx_pu))

        # Normalize weights to probabilities
        probs = weights / weights.sum()

        # Generate synthetic samples
        synthetic_rows = []

        for _ in range(self.n_synthetic_):
            # Weighted random selection of parent from PU
            local_idx = rng.choice(len(idx_pu), p=probs)
            parent_global_idx = idx_pu[local_idx]
            parent_row = X.iloc[parent_global_idx]

            # Use same-type neighbors (same Y and S) per authors' implementation
            neighbors = same_type_neighbors_list[local_idx]

            if len(neighbors) == 0:
                # Edge case: outlier with no same-type neighbors, duplicate parent
                synthetic_rows.append(parent_row.to_dict())
                continue

            # Select random neighbor for interpolation
            neighbor_idx = rng.choice(neighbors)
            neighbor_row = X.iloc[neighbor_idx]

            # SMOTE interpolation
            r = rng.random()
            synth_row = {}

            for col in X.columns:
                if col == self.sens_attr:
                    # Preserve sensitive attribute from parent
                    synth_row[col] = parent_row[col]
                elif col in numeric_cols:
                    # Linear interpolation
                    val = self._interpolate_smote(
                        float(parent_row[col]),
                        float(neighbor_row[col]),
                        r
                    )
                    if pd.api.types.is_integer_dtype(schema['dtypes'][col]):
                        val = round(val)
                    synth_row[col] = val
                else:
                    # Categorical: random selection
                    synth_row[col] = rng.choice([parent_row[col], neighbor_row[col]])

            synthetic_rows.append(synth_row)

        # Combine original and synthetic data
        if synthetic_rows:
            synth_df = pd.DataFrame(synthetic_rows)
            self._enforce_schema(synth_df, schema)
            X_resampled = pd.concat([X, synth_df], ignore_index=True)
            y_resampled = np.concatenate([y, [pos_class] * len(synthetic_rows)])
        else:
            X_resampled = X.copy()
            y_resampled = y.copy()

        return X_resampled, y_resampled
