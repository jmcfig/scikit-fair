"""Loader for the German Credit dataset (Statlog)."""

from importlib.resources import files
from typing import Tuple, Union

import pandas as pd
from sklearn.utils import Bunch

COLUMN_NAMES = [
    "status",
    "month",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment",
    "investment_as_income_percentage",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "installment_plans",
    "housing",
    "number_of_credits",
    "skill_level",
    "people_liable_for",
    "telephone",
    "foreign_worker",
    "credit",
]


def load_german(
    *,
    return_X_y: bool = True,
    as_frame: bool = True,
    preprocessed: bool = True,
    target_column: str = "credit",
) -> Union[Bunch, Tuple[pd.DataFrame, pd.Series]]:
    """
    Load the German Credit dataset from the installed package data.

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
        encoded in place (male=1, female=0) and kept in X.
    target_column : str, default="credit"
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
    data_path = files("skfair.datasets") / "data" / "statlog+german" / "german.data"

    df = pd.read_csv(data_path, sep=" ", header=None, names=COLUMN_NAMES)

    male_codes = {"A91", "A93", "A94"}
    df = df.copy()
    df["sex"] = df["personal_status"].apply(
        lambda v: "male" if v in male_codes else "female"
    )
    df = df.drop(columns=["personal_status"])

    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in dataset columns.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    if preprocessed:
        from ._preprocessing import preprocess_frame
        X = X.copy()
        X["sex"] = X["sex"].map({"male": 1, "female": 0})
        y = y.map({1: 1, 2: 0})  # 1=good credit (favorable), 2=bad credit (unfavorable)
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
        DESCR="German Credit dataset (Statlog). Predict creditworthiness. Sensitive column example: sex.",
    )
