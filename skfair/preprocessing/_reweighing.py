"""Reweighing algorithm (Kamiran & Calders, 2012).

Assigns instance weights to reduce dependence between a sensitive
attribute and the label.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator


class Reweighing(BaseEstimator):
    """
    Reweighing preprocessing technique (Kamiran & Calders, 2012).

    The Reweighing method assigns weights to training samples such that
    the weighted dataset exhibits less statistical dependence between
    a sensitive attribute and the label. It does *not* modify the feature
    matrix or the labels; instead, it outputs instance weights that can
    be passed to downstream learning algorithms via `sample_weight`.

    Conceptual Summary
    ------------------
    Let A be a sensitive attribute (e.g., sex, race) and Y the binary label.
    The method computes the expected probability of each (A, Y) combination
    under independence:

        P(A=a, Y=y)_expected = P(A=a) * P(Y=y)

    and compares it to the empirical joint probabilities:

        P(A=a, Y=y)_observed

    Each sample receives a weight:

        weight = P_expected / P_observed

    so that, after weighting, the distribution of outcomes is closer to
    independence between A and Y.

    Reference
    ---------
    F. Kamiran and T. Calders,
    "Data Preprocessing Techniques for Classification without
    Discrimination," Knowledge and Information Systems, 2012.

    Parameters
    ----------
    sens_attr : str
        Name of the sensitive attribute column in X (pandas DataFrame).

    priv_group : int or str, default=1
        Value in `sens_attr` treated as the privileged group.
        (Not used directly in the weighting formula, but stored
        for consistency with other preprocessors.)

    pos_label : int or str, default=1
        Label considered the favorable outcome.

    Attributes (after fit/fit_transform)
    ------------------------------------
    group_probs_ : dict
        Empirical probabilities P(A=a) and P(Y=y).
        Keys: ("A", a) and ("Y", y).

    joint_probs_ : dict
        Empirical joint probabilities P(A=a, Y=y).
        Keys: (a, y).

    expected_probs_ : dict
        Expected independence probabilities P(A=a)*P(Y=y).
        Keys: (a, y).

    weight_table_ : dict
        Mapping (a, y) -> W(a, y) according to Algorithm 3.

    weights_ : pandas Series of shape (n_samples,)
        Per-sample weights for the data passed to `fit_transform`.
    """

    def __init__(self, sens_attr=None, priv_group=1, pos_label=1):
        self.sens_attr = sens_attr
        self.priv_group = priv_group
        self.pos_label = pos_label

    def _check_inputs(self, X, y_series):
        """Basic validation: DataFrame, sens_attr present, binary labels."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Reweighing currently expects X to be a pandas DataFrame.")

        if self.sens_attr is None:
            raise ValueError(
                "sens_attr must be set to the name of the sensitive attribute column."
            )

        if self.sens_attr not in X.columns:
            raise ValueError(f"Protected attribute '{self.sens_attr}' not found in X columns.")

        unique_labels = pd.unique(y_series)
        if len(unique_labels) != 2:
            raise ValueError(
                f"Reweighing requires binary labels, got {len(unique_labels)} unique labels: {unique_labels}"
            )
        if self.pos_label not in unique_labels:
            raise ValueError(
                f"pos_label={self.pos_label!r} not found in labels {unique_labels}."
            )

        # Negative label is the other one
        neg_label = next(l for l in unique_labels if l != self.pos_label)
        return neg_label

    def fit(self, X, y):
        """
        Fit the reweighing model.

        This method validates the inputs and records basic label
        information (e.g., classes_). It does not compute weights;
        weight computation happens in `fit_transform`.

        Parameters
        ----------
        X : pandas DataFrame
            Must contain `self.sens_attr`.

        y : array-like or pandas Series
            Binary labels.

        Returns
        -------
        self
        """
        # Align y as Series with X index
        if isinstance(y, pd.Series):
            y_series = y.copy()
            y_series.index = X.index
        else:
            y_series = pd.Series(y, index=X.index)

        neg_label = self._check_inputs(X, y_series)
        self.classes_ = (neg_label, self.pos_label)

        return self

    def fit_transform(self, X, y):
        """
        Run Algorithm 3 (Reweighing) and return per-sample weights.

        Algorithm 3 (simplified)
        ------------------------
        For each group value s and class label c:

            W(s, c) :=
                |{X | S = s}| * |{X | Y = c}|
                ---------------------------------
                  |D| * |{X | S = s, Y = c}|

        Then each instance X_i with sensitive attribute S_i and label Y_i
        receives weight W(S_i, Y_i).

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix. Not modified by reweighing.

        y : array-like or pandas Series
            Binary labels. Not modified by reweighing.

        Returns
        -------
        X_out : pandas DataFrame
            Same as X, passed through unchanged.

        weights : pandas Series of shape (n_samples,)
            The reweighing weights corresponding to each instance.
        """
        # Align y as Series with X index
        if isinstance(y, pd.Series):
            y_series = y.copy()
            y_series.index = X.index
        else:
            y_series = pd.Series(y, index=X.index)

        # Validate + set classes_, mirroring Massaging's use of fit
        self.fit(X, y_series)

        S = X[self.sens_attr]
        Y = y_series
        n = len(X)

        groups = pd.unique(S)
        labels = pd.unique(Y)

        group_counts = S.value_counts()
        label_counts = Y.value_counts()
        joint_counts = pd.crosstab(S, Y)

        self.group_probs_ = {}
        self.joint_probs_ = {}
        self.expected_probs_ = {}
        self.weight_table_ = {}

        # P(A=a) and P(Y=y)
        for s in groups:
            self.group_probs_[("A", s)] = group_counts[s] / n
        for c in labels:
            self.group_probs_[("Y", c)] = label_counts[c] / n

        # W(s, c) and related probabilities
        for s in groups:
            for c in labels:
                n_s = group_counts[s]
                n_c = label_counts[c]
                n_s_c = joint_counts.loc[s, c] if (s in joint_counts.index and c in joint_counts.columns) else 0

                if n_s_c > 0:
                    p_obs = n_s_c / n
                    p_a = n_s / n
                    p_y = n_c / n
                    p_exp = p_a * p_y
                    w = p_exp / p_obs
                else:
                    p_obs = 0.0
                    p_a = n_s / n
                    p_y = n_c / n
                    p_exp = p_a * p_y
                    w = 0.0  # no samples with this combination

                self.joint_probs_[(s, c)] = p_obs
                self.expected_probs_[(s, c)] = p_exp
                self.weight_table_[(s, c)] = w

        # Assign weights per instance using W(S_i, Y_i)
        weights = []
        for idx in X.index:
            s_i = S.loc[idx]
            y_i = Y.loc[idx]
            w_i = self.weight_table_[(s_i, y_i)]
            weights.append(w_i)

        self.weights_ = pd.Series(weights, index=X.index)

        return X, self.weights_
