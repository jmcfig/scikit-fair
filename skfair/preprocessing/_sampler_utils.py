"""
Shared utilities for fairness-aware samplers.

Common patterns for DataFrame validation, KNN neighbor finding,
synthetic sample generation, and dtype preservation.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def validate_sampler_input(X, sens_attr):
    """
    Validate DataFrame input and sensitive attribute.

    Parameters
    ----------
    X : any
        Input to validate (should be pandas DataFrame).
    sens_attr : str
        Name of the sensitive attribute column.

    Raises
    ------
    TypeError
        If X is not a pandas DataFrame.
    ValueError
        If sens_attr is not in X.columns.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input X must be a pandas DataFrame.")
    if sens_attr not in X.columns:
        raise ValueError(f"Protected attribute '{sens_attr}' not found in X.")


def extract_numeric_schema(X, exclude_cols=None, compute_bounds=False):
    """
    Extract schema information from DataFrame for type preservation.

    Parameters
    ----------
    X : pandas DataFrame
        Input data.
    exclude_cols : list of str, optional
        Columns to exclude from numeric_cols (e.g., sensitive attribute).
    compute_bounds : bool, default=False
        Whether to compute min/max bounds for numeric columns.

    Returns
    -------
    dict with keys:
        - 'dtypes': dict mapping column name to dtype
        - 'numeric_cols': list of numeric column names
        - 'bounds': dict mapping col -> (min, max), only if compute_bounds=True
    """
    exclude_cols = exclude_cols or []

    dtypes = X.dtypes.to_dict()
    all_numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in all_numeric if c not in exclude_cols]

    result = {
        'dtypes': dtypes,
        'numeric_cols': numeric_cols,
    }

    if compute_bounds:
        result['bounds'] = {
            col: (X[col].min(), X[col].max())
            for col in numeric_cols
        }

    return result


def fit_knn(X_numeric, k_neighbors):
    """
    Fit a KNN model with edge case handling.

    Parameters
    ----------
    X_numeric : ndarray of shape (n_samples, n_features)
        Numeric feature matrix for KNN.
    k_neighbors : int
        Desired number of neighbors.

    Returns
    -------
    nn : NearestNeighbors or None
        Fitted KNN model, or None if too few samples.
    k : int
        Adjusted k value (may be less than k_neighbors).
    """
    n_samples = len(X_numeric)

    if n_samples < 2:
        return None, 0

    k = min(k_neighbors, n_samples - 1)
    if k < 1:
        return None, 0

    nn = NearestNeighbors(n_neighbors=k + 1)  # +1 to account for self
    nn.fit(X_numeric)

    return nn, k


def query_neighbors(nn, X_numeric, idx, n_select=1, rng=None):
    """
    Query neighbors for a sample, excluding self.

    Parameters
    ----------
    nn : NearestNeighbors
        Fitted KNN model.
    X_numeric : ndarray
        Feature matrix (same used to fit nn).
    idx : int
        Index of the query sample in X_numeric.
    n_select : int, default=1
        Number of neighbors to randomly select from result.
    rng : RandomState or None
        Random state for neighbor selection.

    Returns
    -------
    neighbor_indices : ndarray
        Array of selected neighbor indices (length = n_select).
        Returns array of idx repeated if not enough neighbors.
    """
    _, all_neighbors = nn.kneighbors(X_numeric[idx].reshape(1, -1))
    neighbors = all_neighbors[0][1:]  # Exclude self (first result)

    if len(neighbors) == 0:
        return np.array([idx] * n_select)

    if len(neighbors) < n_select:
        # Not enough neighbors, sample with replacement
        if rng is not None:
            return rng.choice(neighbors, size=n_select, replace=True)
        return np.resize(neighbors, n_select)

    if rng is not None:
        return rng.choice(neighbors, size=n_select, replace=False)

    return neighbors[:n_select]


def enforce_dtypes(df, dtypes_dict):
    """
    Best-effort dtype restoration on DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to modify (in place).
    dtypes_dict : dict
        Mapping of column name to desired dtype.
    """
    for col, dtype in dtypes_dict.items():
        if col not in df.columns:
            continue
        try:
            df[col] = df[col].astype(dtype)
        except (ValueError, TypeError):
            pass


def interpolate_smote(parent_val, neighbor_val, r):
    """
    SMOTE-style linear interpolation.

    Formula: S = B + r * (N - B)

    Parameters
    ----------
    parent_val : float
        Base/parent value.
    neighbor_val : float
        Neighbor value.
    r : float
        Random value in [0, 1].

    Returns
    -------
    float
        Interpolated value.
    """
    return parent_val + r * (neighbor_val - parent_val)


def interpolate_de(parent_val, n1_val, n2_val, f):
    """
    Differential Evolution style mutation.

    Formula: S = P + f * (N1 - N2)

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
    return parent_val + f * (n1_val - n2_val)


def generate_synthetic_value(col, base_row, neighbor_rows, schema,
                             interpolation='smote', rng=None,
                             f=0.8, clip_bounds=None):
    """
    Generate a synthetic value for a single column.

    Parameters
    ----------
    col : str
        Column name.
    base_row : Series
        Parent/base sample.
    neighbor_rows : list of Series
        Neighbor samples (1 for SMOTE, 2 for DE).
    schema : dict
        Schema from extract_numeric_schema().
    interpolation : str, default='smote'
        'smote' for SMOTE interpolation, 'de' for Differential Evolution.
    rng : RandomState or None
        Random state.
    f : float, default=0.8
        Mutation factor for DE interpolation.
    clip_bounds : dict or None
        Optional bounds for clipping numeric values.

    Returns
    -------
    Synthetic value for the column.
    """
    base_val = base_row[col]
    dtypes = schema['dtypes']
    numeric_cols = schema['numeric_cols']

    if col in numeric_cols:
        # Numeric interpolation
        if interpolation == 'smote':
            r = rng.random() if rng else np.random.random()
            new_val = interpolate_smote(float(base_val), float(neighbor_rows[0][col]), r)
        else:  # de
            new_val = interpolate_de(
                float(base_val),
                float(neighbor_rows[0][col]),
                float(neighbor_rows[1][col]),
                f
            )

        # Clip if bounds provided
        if clip_bounds and col in clip_bounds:
            low, high = clip_bounds[col]
            new_val = np.clip(new_val, low, high)

        # Round if integer dtype
        if pd.api.types.is_integer_dtype(dtypes[col]):
            new_val = round(new_val)

        return new_val

    else:
        # Categorical/other: random selection
        choices = [base_val] + [nr[col] for nr in neighbor_rows]
        if rng is not None:
            return rng.choice(choices)
        return np.random.choice(choices)
