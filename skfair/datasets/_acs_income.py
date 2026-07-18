"""Loader for the ACSIncome (American Community Survey) dataset."""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

# ACSIncome (Ding et al., 2021), raw 2018 1-year PUMS release on OpenML.
_OPENML_DATA_ID = 43141
# Income threshold defining the binary target, as in the original task.
_INCOME_THRESHOLD = 50_000


def fetch_acs_income(
    *,
    return_X_y: bool = True,
    as_frame: bool = True,
    preprocessed: bool = True,
    subsample: Optional[int] = None,
    random_state: Optional[int] = None,
    data_home: Optional[str] = None,
) -> Union[Bunch, Tuple[pd.DataFrame, pd.Series]]:
    """
    Fetch the ACSIncome dataset (Ding et al., 2021) from OpenML.

    ACSIncome was introduced as a large-scale alternative to the Adult
    dataset, compiled from the American Community Survey (ACS) Public Use
    Microdata Sample (PUMS). This loader retrieves the 2018 1-year release
    covering all US states and Puerto Rico (1,664,500 rows). The task is
    predicting whether a person's income exceeds $50,000.

    Unlike the other datasets in :mod:`skfair.datasets`, which ship bundled
    with the package, ACSIncome is too large to distribute and is fetched
    on demand from OpenML (dataset ID 43141). The download happens once and
    is cached locally by scikit-learn (see ``data_home``).

    Parameters
    ----------
    return_X_y : bool, default=True
        If True, returns (X, y). If False, returns a Bunch object.
    as_frame : bool, default=True
        If True, returns pandas objects (DataFrame / Series).
        If False, returns NumPy arrays.
    preprocessed : bool, default=True
        If True, binarize the target (income > $50,000), encode the
        sensitive column ``SEX`` in place (1 = male, 0 = female), keep
        ``RAC1P`` as a multi-valued sensitive column (integer PUMS race
        codes), drop the auxiliary state column ``ST``, and standardize the
        remaining numerical columns.
    subsample : int, optional
        If given, return a random subset with this number of rows, drawn
        with ``random_state``. Useful for tractable demonstrations, since
        the full dataset has over 1.6 million rows.
    random_state : int, optional
        Seed for the ``subsample`` draw.
    data_home : str, optional
        Cache directory for the OpenML download. Defaults to
        scikit-learn's standard location (``~/scikit_learn_data``).

    Returns
    -------
    data : Bunch or (X, y)
        If return_X_y is True, returns ``(X, y)`` where X is a DataFrame
        (or ndarray when as_frame=False) that includes the sensitive
        columns ``SEX`` and ``RAC1P``.

        If return_X_y is False, returns a Bunch with fields:

        - ``data`` : features including sensitive columns (DataFrame or ndarray)
        - ``target`` : target (Series or ndarray)
        - ``frame`` : full DataFrame with features and target
        - ``feature_names`` : list of feature column names
        - ``DESCR`` : short description string
    """
    bunch = fetch_openml(
        data_id=_OPENML_DATA_ID,
        as_frame=True,
        data_home=data_home,
        parser="auto",
    )
    df = bunch.frame

    y = df["PINCP"]
    X = df.drop(columns=["PINCP"])

    if subsample is not None:
        X = X.sample(n=subsample, random_state=random_state)
        y = y.loc[X.index]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    if preprocessed:
        from ._preprocessing import preprocess_frame
        X = X.copy()
        # PUMS codes: SEX 1 = male, 2 = female; RAC1P race codes 1-9.
        X["SEX"] = np.where(X["SEX"] == 1, 1, 0)
        X["RAC1P"] = X["RAC1P"].astype(int)
        y = np.where(y > _INCOME_THRESHOLD, 1, 0)
        X = X.drop(columns=["ST"])
        X = preprocess_frame(X, exclude_cols=["SEX", "RAC1P"])

    feature_names = list(X.columns)

    # Build frame before any array conversion; y may already be an ndarray
    if isinstance(y, pd.Series):
        frame = pd.concat([X, y], axis=1)
    else:
        frame = pd.concat(
            [X, pd.Series(y, name="PINCP", index=X.index)], axis=1
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
        DESCR=(
            "ACSIncome dataset (Ding et al., 2021), 2018 1-year ACS PUMS. "
            "Predict whether income exceeds $50K/yr. Fetched from OpenML "
            "(ID 43141) and cached locally; not bundled with the package. "
            "Sensitive column examples: SEX (binary), RAC1P (multi-valued)."
        ),
    )
