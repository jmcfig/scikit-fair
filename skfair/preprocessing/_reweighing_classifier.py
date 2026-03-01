"""Pipeline-compatible classifier wrapper for Reweighing."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from ._reweighing import Reweighing


class ReweighingClassifier(BaseEstimator, ClassifierMixin):
    """
    Meta-estimator that combines Reweighing with any classifier.

    This wrapper makes Reweighing compatible with sklearn Pipelines by
    encapsulating the weight computation and passing weights to the
    underlying classifier's `fit` method via `sample_weight`.

    The workflow is:
    1. fit(X, y): Compute fairness weights using Reweighing, then fit the
       classifier with those weights.
    2. predict(X): Delegate to the fitted classifier.

    Parameters
    ----------
    estimator : classifier, default=LogisticRegression()
        The classifier to train with reweighted samples.
        Must support the `sample_weight` parameter in `fit()`.

    sens_attr : str
        Name of the sensitive attribute column in X.

    priv_group : int or str, default=1
        Value in `sens_attr` treated as the privileged group.

    pos_label : int or str, default=1
        Label considered the favorable outcome.

    Attributes
    ----------
    reweigher_ : Reweighing
        Fitted Reweighing instance.

    estimator_ : classifier
        Fitted classifier instance.

    classes_ : ndarray
        Class labels from the fitted classifier.

    weights_ : pandas Series
        Sample weights computed during fit.

    Example
    -------
    >>> from skfair.preprocessing import ReweighingClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> clf = ReweighingClassifier(
    ...     estimator=RandomForestClassifier(),
    ...     sens_attr='sex',
    ...     priv_group=1,
    ...     pos_label=1
    ... )
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        estimator=None,
        sens_attr=None,
        priv_group=1,
        pos_label=1,
    ):
        self.estimator = estimator
        self.sens_attr = sens_attr
        self.priv_group = priv_group
        self.pos_label = pos_label

    def fit(self, X, y):
        """
        Fit the reweighing classifier.

        1. Compute fairness weights using Reweighing
        2. Fit the underlying classifier with sample_weight

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix. Must contain `sens_attr` column.

        y : array-like
            Binary target labels.

        Returns
        -------
        self
        """
        # Create reweigher with our parameters
        self.reweigher_ = Reweighing(
            sens_attr=self.sens_attr,
            priv_group=self.priv_group,
            pos_label=self.pos_label,
        )

        # Compute weights
        _, self.weights_ = self.reweigher_.fit_transform(X, y)

        # Clone the estimator (or use default)
        if self.estimator is None:
            self.estimator_ = LogisticRegression()
        else:
            self.estimator_ = clone(self.estimator)

        # Fit with sample weights
        self.estimator_.fit(X, y, sample_weight=self.weights_)

        # Store classes for sklearn compatibility
        self.classes_ = self.estimator_.classes_

        return self

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix.

        Returns
        -------
        y_pred : ndarray
            Predicted class labels.
        """
        check_is_fitted(self, ["estimator_", "reweigher_"])
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Only available if the underlying estimator supports predict_proba.

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self, ["estimator_", "reweigher_"])
        return self.estimator_.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        """
        Return accuracy score on the given test data.

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix.

        y : array-like
            True labels.

        sample_weight : array-like, optional
            Sample weights for scoring.

        Returns
        -------
        score : float
            Accuracy score.
        """
        check_is_fitted(self, ["estimator_", "reweigher_"])
        return self.estimator_.score(X, y, sample_weight=sample_weight)
