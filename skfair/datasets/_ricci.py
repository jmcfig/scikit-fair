"""Loader for the Ricci v. DeStefano firefighter promotions dataset."""

from importlib.resources import files
from typing import Tuple, Union

import pandas as pd
import numpy as np
from sklearn.utils import Bunch


def load_ricci(
    *,
    return_X_y: bool = True,
    as_frame: bool = True,
    preprocessed: bool = True,
    target_column: str = "Combine",
) -> Union[Bunch, Tuple[pd.DataFrame, pd.Series]]:
    """
    Load the Ricci v. DeStefano firefighter promotions dataset.

    Parameters
    ----------
    return_X_y : bool, default=True
        If True, returns (X, y). If False, returns a Bunch object.
    as_frame : bool, default=True
        If True, returns pandas objects (DataFrame / Series).
        If False, returns NumPy arrays.
    preprocessed : bool, default=True
        If True, encode categorical columns as numeric:
        Race: W=1, else=0; Position: Captain=1, Lieutenant=0.
    target_column : str, default="Combine"
        Name of the target column in the CSV.

    Returns
    -------
    data : Bunch or (X, y)
        If return_X_y is True, returns ``(X, y)`` where X is a DataFrame
        (or ndarray when as_frame=False) that includes the sensitive columns.

        If return_X_y is False, returns a Bunch with fields:

        - ``data`` : features including sensitive columns (DataFrame or ndarray)
        - ``target`` : target (Series or ndarray)
        - ``frame`` : full DataFrame with features and target
        - ``feature_names`` : list of feature column names
        - ``DESCR`` : short description string
    """
    data_path = files("skfair.datasets") / "data" / "Ricci.csv"

    df = pd.read_csv(data_path)

    # Drop the row-index column
    df = df.drop(columns=["rownames"], errors="ignore")

    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in dataset columns.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    if preprocessed:
        from ._preprocessing import preprocess_frame
        X = X.copy()
        X["Race"] = np.where(X["Race"] == "W", 1, 0)
        X["Position"] = np.where(X["Position"] == "Captain", 1, 0)
        y = (y >= 70).astype(int)


    feature_names = list(X.columns)

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
        DESCR="Ricci v. DeStefano firefighter promotions dataset. Predict combined test score. Sensitive columns: Race"
        ".",
    )
