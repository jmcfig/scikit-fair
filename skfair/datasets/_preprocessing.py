"""Shared preprocessing utilities for dataset loaders."""

import pandas as pd


def preprocess_frame(X: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    """One-hot encode categorical columns and standardize numerical columns.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with mixed column types.
    exclude_cols : list of str, optional
        Columns to pass through unchanged (not standardized, not one-hot
        encoded). Intended for binary-encoded sensitive attributes whose
        0/1 values must remain intact.

    Returns
    -------
    pd.DataFrame
        Fully numeric DataFrame with standardized numericals followed by
        one-hot encoded categoricals (``drop_first=True``), with any
        excluded columns appended at the end as-is.
    """
    exclude_cols = list(exclude_cols or [])
    passthrough = X[exclude_cols].reset_index(drop=True) if exclude_cols else None
    X = X.drop(columns=exclude_cols, errors="ignore")

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns

    # Standardize numerical columns
    X_num = X[num_cols].astype(float)
    means = X_num.mean()
    stds = X_num.std()
    stds = stds.replace(0, 1)  # avoid division by zero
    X_num = (X_num - means) / stds

    # One-hot encode categorical columns
    if len(cat_cols) > 0:
        X_cat = pd.get_dummies(X[cat_cols], drop_first=True)
        result = pd.concat([X_num, X_cat], axis=1)
    else:
        result = X_num

    if passthrough is not None:
        result = pd.concat([result, passthrough], axis=1)

    return result
