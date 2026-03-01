"""
Fairway algorithm for removing ambiguous data points.

Removes samples that cause conflicting predictions between
models trained on privileged vs unprivileged groups.
"""

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from ._base import BaseFairSampler


class FairwayRemover(BaseFairSampler):
    """
    Removes 'ambiguous' data points that cause conflicting predictions
    between privileged and unprivileged group models.

    Trains two separate models on privileged and unprivileged groups,
    then removes any samples where the models disagree on predictions.

    Parameters
    ----------
    sens_attr : str
        Name of the sensitive attribute column in X.

    priv_group : int or str
        Value in `sens_attr` that represents the privileged group.

    estimator : sklearn estimator or None, default=None
        Base estimator to use for training group-specific models.
        If None, uses LogisticRegression(solver='liblinear').

    Attributes
    ----------
    model_p_ : estimator
        Model trained on the privileged group.

    model_u_ : estimator
        Model trained on the unprivileged group.
    """

    _sampling_type = "clean-sampling"

    def __init__(self, sens_attr, priv_group, estimator=None):
        super().__init__(sens_attr, random_state=None)
        self.priv_group = priv_group
        self.estimator = estimator

    def _fit_resample(self, X, y):
        """
        Remove ambiguous samples where group-specific models disagree.
        """
        # 1. Setup Models
        if self.estimator is None:
            self.model_p_ = LogisticRegression(solver='liblinear')
            self.model_u_ = LogisticRegression(solver='liblinear')
        else:
            self.model_p_ = clone(self.estimator)
            self.model_u_ = clone(self.estimator)

        # 2. Split Data by Protected Attribute
        mask_priv = (X[self.sens_attr] == self.priv_group)
        mask_unpriv = ~mask_priv

        # Handle y indexing safely
        y_p = y[mask_priv] if isinstance(y, (pd.Series, pd.DataFrame)) else y[mask_priv.values]
        y_u = y[mask_unpriv] if isinstance(y, (pd.Series, pd.DataFrame)) else y[mask_unpriv.values]

        # 3. Train Sub-Models
        X_p = X.loc[mask_priv].drop(columns=[self.sens_attr])
        X_u = X.loc[mask_unpriv].drop(columns=[self.sens_attr])

        self.model_p_.fit(X_p, y_p)
        self.model_u_.fit(X_u, y_u)

        # 4. Detect Ambiguity (Bias)
        X_pred = X.drop(columns=[self.sens_attr])

        pred_p = self.model_p_.predict(X_pred)
        pred_u = self.model_u_.predict(X_pred)

        mask_keep = (pred_p == pred_u)

        # 5. Filter and Return
        X_res = X.loc[mask_keep].copy()

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_res = y.loc[X.index[mask_keep]].copy()
        else:
            y_res = y[mask_keep].copy()

        return X_res, y_res
