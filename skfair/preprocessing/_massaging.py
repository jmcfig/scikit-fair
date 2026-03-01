"""
Massaging algorithm (Kamiran & Calders, 2012).

Modifies class labels to reduce discrimination in the training data.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from ._base import BaseFairSampler


class Massaging(BaseFairSampler):
    """
    Massaging preprocessing technique (Kamiran & Calders, 2012).

    Modifies class labels of training samples to reduce the statistical
    dependence between the sensitive attribute and the label. Identifies
    candidates for promotion (unprivileged with negative label) and
    demotion (privileged with positive label), then swaps their labels.

    Parameters
    ----------
    sens_attr : str
        Name of the sensitive attribute column in X.

    priv_group : int or str, default=1
        Value in `sens_attr` that represents the privileged group.

    pos_label : int or str, default=1
        Value representing the positive/favorable outcome.

    estimator : sklearn estimator or None, default=None
        Estimator used to rank candidates for label swapping.
        Must support `predict_proba`. If None, uses LogisticRegression.

    Attributes
    ----------
    ranker_ : estimator
        Fitted ranker used to prioritize candidates.

    classes_ : ndarray
        Class labels from the ranker.
    """

    _sampling_type = "clean-sampling"

    def __init__(self, sens_attr=None, priv_group=1, pos_label=1, estimator=None):
        super().__init__(sens_attr, random_state=None)
        self.priv_group = priv_group
        self.pos_label = pos_label
        self.estimator = estimator

    def _check_X_y(self, X, y):
        """Override to add sens_attr None check before base validation."""
        if self.sens_attr is None:
            raise ValueError("sens_attr parameter must be specified.")
        return super()._check_X_y(X, y)

    def _fit_resample(self, X, y):
        """
        Massage the dataset by swapping labels to reduce discrimination.
        """
        self._check_binary_labels(y)

        if self.estimator is None:
            self.ranker_ = LogisticRegression(solver="liblinear")
        else:
            self.ranker_ = clone(self.estimator)

        if not hasattr(self.ranker_, "predict_proba"):
            raise ValueError("The provided estimator must support `predict_proba`.")

        self.ranker_.fit(X, y)
        self.classes_ = self.ranker_.classes_

        priv_mask = X[self.sens_attr] == self.priv_group
        unpriv_mask = ~priv_mask

        p_priv = np.mean(y[priv_mask] == self.pos_label)
        p_unpriv = np.mean(y[unpriv_mask] == self.pos_label)
        disc_score = p_priv - p_unpriv

        if disc_score <= 0:
            return X.copy(), y.copy()

        n_priv = priv_mask.sum()
        n_unpriv = unpriv_mask.sum()
        n = len(X)

        m_swap = disc_score * n_priv * n_unpriv / n
        m_swap = int(round(m_swap))

        if m_swap <= 0:
            return X.copy(), y.copy()

        pr_candidates, dem_candidates = self._rank_candidates(X, y, priv_mask, unpriv_mask)
        m_swap = min(m_swap, len(pr_candidates), len(dem_candidates))

        y_mod = pd.Series(y, index=X.index).copy()
        neg_label = self._get_neg_label(y)

        # Use .loc since _rank_candidates returns .index values
        y_mod.loc[pr_candidates[:m_swap]] = self.pos_label
        y_mod.loc[dem_candidates[:m_swap]] = neg_label
        y_mod = y_mod.values

        return X.copy(), y_mod

    def _rank_candidates(self, X, y, priv_mask, unpriv_mask):
        """Rank promotion and demotion candidates by predicted probability."""
        pr_mask = unpriv_mask & (y != self.pos_label)
        dem_mask = priv_mask & (y == self.pos_label)

        try:
            pos_idx = list(self.classes_).index(self.pos_label)
        except ValueError:
            raise ValueError(f"pos_label {self.pos_label} not in classes {self.classes_}")

        scores_all = self.ranker_.predict_proba(X)[:, pos_idx]
        scores_series = pd.Series(scores_all, index=X.index)

        pr_sorted = scores_series[pr_mask].sort_values(ascending=False).index
        dem_sorted = scores_series[dem_mask].sort_values(ascending=True).index

        return pr_sorted, dem_sorted

    def _check_binary_labels(self, y):
        """Validate that labels are binary and pos_label is present."""
        unique = pd.unique(y)
        if len(unique) != 2:
            raise ValueError(f"Massaging requires binary labels, got {len(unique)}.")
        if self.pos_label not in unique:
            raise ValueError(f"pos_label={self.pos_label} not found in data.")

    def _get_neg_label(self, y):
        """Get the negative label (the one that isn't pos_label)."""
        unique = pd.unique(y)
        for lab in unique:
            if lab != self.pos_label:
                return lab
        raise RuntimeError("Could not determine negative label.")
