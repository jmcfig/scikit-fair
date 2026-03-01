"""Loader for the Adult (Census Income) dataset."""

from importlib.resources import files
from typing import Tuple, Union

import pandas as pd
import numpy as np
from sklearn.utils import Bunch


def load_adult(
    *,
    return_X_y: bool = True,
    as_frame: bool = True,
    preprocessed: bool = True,
    target_column: str = "Probability",
) -> Union[Bunch, Tuple[pd.DataFrame, pd.Series]]:
    """
    Load the Adult dataset from the installed package data.

    Parameters
    ----------
    return_X_y : bool, default=True
        If True, returns (X, y). If False, returns a Bunch object.
    as_frame : bool, default=True
        If True, returns pandas objects (DataFrame / Series).
        If False, returns NumPy arrays.
    preprocessed : bool, default=True
        If True, one-hot encode categorical columns and standardize
        numerical columns before returning. The sensitive column is
        encoded in place and kept in X.
    target_column : str, default="Probability"
        Name of the target column in the CSV.

    Returns
    -------
    data : Bunch or (X, y)
        If return_X_y is True, returns ``(X, y)`` where X is a DataFrame
        (or ndarray when as_frame=False) that includes the sensitive column.

        If return_X_y is False, returns a Bunch with fields:

        - ``data`` : features including sensitive column (DataFrame or ndarray)
        - ``target`` : target (Series or ndarray)
        - ``frame`` : full DataFrame with features and target
        - ``feature_names`` : list of feature column names
        - ``DESCR`` : short description string
    """
    data_path = files("skfair.datasets") / "data" / "adult.csv"

    df = pd.read_csv(data_path)

    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in dataset columns.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    if preprocessed:
        from ._preprocessing import preprocess_frame
        X = X.copy()
        X["sex"] = np.where(X["sex"] == " Male", 1, 0)
        X["race"] = np.where(X["race"] == " White", 1, 0)
        y = np.where(y == " >50K", 1, 0)
        X = X.drop(columns=["native-country", "fnlwgt"])
        X = preprocess_frame(X, exclude_cols=["sex", "race"])

    feature_names = list(X.columns)

    # Build frame before any array conversion; y may already be an ndarray
    if isinstance(y, pd.Series):
        frame = pd.concat([X, y], axis=1)
    else:
        frame = pd.concat(
            [X, pd.Series(y, name=target_column, index=X.index)], axis=1
        )

    if not as_frame:
        X_out = X.to_numpy()
        y_out = y if not isinstance(y, pd.Series) else y.to_numpy()
    else:
        X_out = X
        y_out = y

    if return_X_y:
        return X_out, y_out

    return Bunch(
        data=X_out,
        target=y_out,
        frame=frame,
        feature_names=feature_names,
        DESCR="Adult census income dataset. Predict whether income exceeds $50K/yr. Sensitive column example: sex.",
    )
