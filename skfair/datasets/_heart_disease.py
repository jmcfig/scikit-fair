"""Loader for the Heart Disease dataset (Statlog)."""

from importlib.resources import files
from typing import Tuple, Union

import pandas as pd
from sklearn.utils import Bunch

COLUMN_NAMES = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "serum_cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_induced_angina",
    "oldpeak",
    "slope",
    "num_major_vessels",
    "thal",
    "heart_disease",
]


def load_heart_disease(
    *,
    return_X_y: bool = True,
    as_frame: bool = True,
    preprocessed: bool = True,
    target_column: str = "heart_disease",
) -> Union[Bunch, Tuple[pd.DataFrame, pd.Series]]:
    """
    Load the Heart Disease dataset (Statlog) from the installed package data.

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
    target_column : str, default="heart_disease"
        Name of the target column.

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
    data_path = files("skfair.datasets") / "data" / "statlog+heart" / "heart.dat"

    df = pd.read_csv(data_path, sep=" ", header=None, names=COLUMN_NAMES)

    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in dataset columns.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    if preprocessed:
        from ._preprocessing import preprocess_frame
        X = X.copy()
        X["sex"] = X["sex"].astype(int)
        y = y.map({1: 1, 2: 0})  # 1=no disease (favorable), 2=disease present (unfavorable)
        y.name = "target"
        X = preprocess_frame(X, exclude_cols=["sex"])

    feature_names = list(X.columns)

    # Build frame before any array conversion
    frame = pd.concat([X, y], axis=1)

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
        DESCR="Heart Disease dataset (Statlog). Predict presence of heart disease. Sensitive column: sex.",
    )
