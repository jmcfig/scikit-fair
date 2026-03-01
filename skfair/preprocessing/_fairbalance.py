"""FairBalance algorithm (Yu, Chakraborty & Menzies, 2024).

Assigns instance weights to equalise class distributions within
each demographic group.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator


class FairBalance(BaseEstimator):
    """
    FairBalance preprocessing technique (Yu, Chakraborty & Menzies, 2024).

    FairBalance assigns weights to training samples to achieve equalized odds
    by balancing the class distribution within each demographic group. It does
    *not* modify the feature matrix or labels; instead, it outputs instance
    weights that can be passed to downstream learning algorithms via
    `sample_weight`.

    Conceptual Summary
    ------------------
    The key insight is that violation of equalized odds is caused by different
    class distributions across demographic groups. FairBalance fixes this by
    weighting samples so each (sensitive group, class) combination has balanced
    influence.

    For each sample with sensitive attribute A=a and label Y=y:

        weight = |A=a| / |A=a, Y=y|

    This ensures the weighted class distribution becomes 1:1 (balanced) within
    each demographic group, which is a sufficient condition for achieving
    smAOD=0 (smoothed maximum Average Odds Difference).

    Variant Mode
    ------------
    When `variant=True`, uses FairBalanceVariant formula:

        weight = 1 / |A=a, Y=y|

    Then rescales all weights so they sum to n_samples. This treats all
    demographic groups equally regardless of their size.

    Reference
    ---------
    Z. Yu, J. Chakraborty, and T. Menzies,
    "FairBalance: How to Achieve Equalized Odds With Data Pre-processing,"
    IEEE Transactions on Software Engineering, 2024.
    https://github.com/hil-se/FairBalance

    Parameters
    ----------
    sens_attr : str
        Name of the sensitive attribute column in X (pandas DataFrame).

    variant : bool, default=False
        If False, use FairBalance formula (preserves group size differences).
        If True, use FairBalanceVariant formula (treats all groups equally).

    pos_label : int or str, default=1
        Label considered the favorable outcome.

    Attributes (after fit/fit_transform)
    ------------------------------------
    classes_ : tuple
        The two class labels (neg_label, pos_label).

    weight_table_ : dict
        Mapping (a, y) -> weight for each (sensitive attribute, label) combination.

    weights_ : pandas Series of shape (n_samples,)
        Per-sample weights for the data passed to `fit_transform`.

    group_counts_ : dict
        Count of samples in each sensitive attribute group. Keys: attribute values.

    joint_counts_ : dict
        Count of samples in each (attribute, label) combination. Keys: (a, y).
    """

    def __init__(self, sens_attr=None, variant=False, pos_label=1):
        self.sens_attr = sens_attr
        self.variant = variant
        self.pos_label = pos_label

    def _check_inputs(self, X, y_series):
        """Basic validation: DataFrame, sens_attr present, binary labels."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FairBalance expects X to be a pandas DataFrame.")

        if self.sens_attr is None:
            raise ValueError(
                "sens_attr must be set to the name of the sensitive attribute column."
            )

        if self.sens_attr not in X.columns:
            raise ValueError(
                f"Protected attribute '{self.sens_attr}' not found in X columns."
            )

        unique_labels = pd.unique(y_series)
        if len(unique_labels) != 2:
            raise ValueError(
                f"FairBalance requires binary labels, got {len(unique_labels)} "
                f"unique labels: {unique_labels}"
            )
        if self.pos_label not in unique_labels:
            raise ValueError(
                f"pos_label={self.pos_label!r} not found in labels {unique_labels}."
            )

        # Negative label is the other one
        neg_label = next(lbl for lbl in unique_labels if lbl != self.pos_label)
        return neg_label

    def fit(self, X, y):
        """
        Fit the FairBalance model.

        This method validates the inputs and records basic label information.
        It does not compute weights; weight computation happens in `fit_transform`.

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
        Compute FairBalance weights and return them with the original data.

        Weight Calculation
        ------------------
        For FairBalance (variant=False):
            w(A=a, Y=y) = |A=a| / |A=a, Y=y|

        For FairBalanceVariant (variant=True):
            w(A=a, Y=y) = 1 / |A=a, Y=y|
            (then rescaled so sum of weights = n_samples)

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix. Not modified by FairBalance.

        y : array-like or pandas Series
            Binary labels. Not modified by FairBalance.

        Returns
        -------
        X_out : pandas DataFrame
            Same as X, passed through unchanged.

        weights : pandas Series of shape (n_samples,)
            The FairBalance weights corresponding to each instance.
        """
        # Check DataFrame early to provide clear error message
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FairBalance expects X to be a pandas DataFrame.")

        if isinstance(y, pd.Series):
            y_series = y.copy()
            y_series.index = X.index
        else:
            y_series = pd.Series(y, index=X.index)

        # Validate and set classes_
        self.fit(X, y_series)

        A = X[self.sens_attr]
        Y = y_series
        n = len(X)

        groups = pd.unique(A)
        labels = pd.unique(Y)

        # Count samples per group and per (group, label) combination
        group_counts = A.value_counts()
        joint_counts = pd.crosstab(A, Y)

        self.group_counts_ = group_counts.to_dict()
        self.joint_counts_ = {}
        self.weight_table_ = {}

        # Compute weights for each (group, label) combination
        for a in groups:
            n_a = group_counts[a]
            for c in labels:
                n_a_c = (
                    joint_counts.loc[a, c]
                    if (a in joint_counts.index and c in joint_counts.columns)
                    else 0
                )

                self.joint_counts_[(a, c)] = n_a_c

                if n_a_c > 0:
                    if self.variant:
                        # FairBalanceVariant: w = 1 / |A=a, Y=y|
                        w = 1.0 / n_a_c
                    else:
                        # FairBalance: w = |A=a| / |A=a, Y=y|
                        w = n_a / n_a_c
                else:
                    w = 0.0

                self.weight_table_[(a, c)] = w

        # Assign weights to each sample
        weights = []
        for idx in X.index:
            a_i = A.loc[idx]
            y_i = Y.loc[idx]
            w_i = self.weight_table_[(a_i, y_i)]
            weights.append(w_i)

        weights = np.array(weights)

        # For variant mode, rescale weights to sum to n_samples
        if self.variant and weights.sum() > 0:
            weights = weights * (n / weights.sum())

        self.weights_ = pd.Series(weights, index=X.index)

        return X, self.weights_
