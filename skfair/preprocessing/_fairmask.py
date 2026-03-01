"""FairMask algorithm (Peng et al., 2021).

Masks sensitive attributes at inference time using an ensemble of
extrapolation models.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from imblearn.over_sampling import SMOTE


class FairMask(BaseEstimator, ClassifierMixin):
    """
    Meta-estimator that masks sensitive attributes at inference time.

    FairMask trains an ensemble of extrapolation models to predict sensitive
    attributes from non-sensitive features. At inference time, it replaces
    actual sensitive attribute values with synthetic values predicted by the
    ensemble via weighted voting. This achieves "procedural fairness" - the
    classifier's decision is based on what non-sensitive features imply,
    not the actual sensitive attribute.

    The workflow is:
    1. fit(X, y): Train extrapolation models (sensitive attr predictors) and
       fit the underlying classifier on original data.
    2. predict(X): Replace sensitive attributes with synthetic values from
       extrapolation models, then delegate to the fitted classifier.

    Parameters
    ----------
    estimator : classifier, default=LogisticRegression()
        The classifier to wrap. Will be trained on original data.

    sens_attr : str
        Name of the sensitive attribute column in X.

    budget : int, default=10
        Number of extrapolation models in the ensemble.

    extrapolation_model : estimator, default=LogisticRegression()
        Model type used for extrapolation (predicting sensitive attribute
        from non-sensitive features). Will be cloned for each model in budget.

    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    estimator_ : classifier
        Fitted classifier instance.

    extrapolation_models_ : list
        List of fitted extrapolation models.

    model_weights_ : ndarray
        Weights for each extrapolation model based on validation accuracy.

    classes_ : ndarray
        Class labels from the fitted classifier.

    Example
    -------
    >>> from skfair.preprocessing import FairMask
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> clf = FairMask(
    ...     estimator=RandomForestClassifier(),
    ...     sens_attr='sex',
    ...     budget=10
    ... )
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)

    References
    ----------
    Peng, K., et al. "FairMask: Better Fairness via Model-based Rebalancing of
    Protected Attributes." arXiv:2110.01109 (2021).
    """

    def __init__(
        self,
        estimator=None,
        sens_attr=None,
        budget=10,
        extrapolation_model=None,
        random_state=None,
    ):
        self.estimator = estimator
        self.sens_attr = sens_attr
        self.budget = budget
        self.extrapolation_model = extrapolation_model
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the FairMask classifier.

        1. Train extrapolation models to predict sensitive attribute
        2. Fit the underlying classifier on original data

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix. Must contain `sens_attr` column.

        y : array-like
            Target labels.

        Returns
        -------
        self
        """
        if self.sens_attr is None:
            raise ValueError("sens_attr must be specified")

        if self.sens_attr not in X.columns:
            raise ValueError(f"Protected attribute '{self.sens_attr}' not found in X")

        # Separate sensitive and non-sensitive features
        P = X[self.sens_attr].values
        NP = X.drop(columns=[self.sens_attr])

        # Train extrapolation models
        self.extrapolation_models_ = []
        self.model_weights_ = []

        rng = np.random.RandomState(self.random_state)

        for _ in range(self.budget):
            # Split for validation to compute weights
            seed = rng.randint(0, 2**31)
            NP_train, NP_val, P_train, P_val = train_test_split(
                NP, P, test_size=0.2, random_state=seed
            )

            # Apply SMOTE to balance sensitive attribute classes
            try:
                smote = SMOTE(random_state=seed)
                NP_resampled, P_resampled = smote.fit_resample(NP_train, P_train)
            except ValueError:
                # If SMOTE fails (e.g., too few samples), use original data
                NP_resampled, P_resampled = NP_train, P_train

            # Clone and fit extrapolation model
            if self.extrapolation_model is None:
                model = LogisticRegression(random_state=seed, max_iter=1000)
            else:
                model = clone(self.extrapolation_model)
                if hasattr(model, 'random_state'):
                    model.random_state = seed

            model.fit(NP_resampled, P_resampled)

            # Calculate weight based on validation accuracy
            weight = model.score(NP_val, P_val)

            self.extrapolation_models_.append(model)
            self.model_weights_.append(weight)

        self.model_weights_ = np.array(self.model_weights_)

        # Normalize weights
        weight_sum = self.model_weights_.sum()
        if weight_sum > 0:
            self.model_weights_ = self.model_weights_ / weight_sum

        # Clone and fit the main classifier on original data
        if self.estimator is None:
            self.estimator_ = LogisticRegression(max_iter=1000)
        else:
            self.estimator_ = clone(self.estimator)

        self.estimator_.fit(X, y)

        # Store classes for sklearn compatibility
        self.classes_ = self.estimator_.classes_

        return self

    def _synthesize_sensitive_attr(self, X):
        """
        Replace sensitive attribute with synthetic values from extrapolation models.

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix with sensitive attribute column.

        Returns
        -------
        X_masked : pandas DataFrame
            Feature matrix with synthetic sensitive attribute.
        """
        NP = X.drop(columns=[self.sens_attr])

        # Get predictions from all extrapolation models
        predictions = np.array([
            model.predict(NP) for model in self.extrapolation_models_
        ])

        # Weighted voting: for each sample, pick the class with the highest
        # total weight across models that predicted it.
        unique_vals = np.unique(predictions)
        n_samples = predictions.shape[1]
        weighted_scores = np.zeros((len(unique_vals), n_samples))
        for k, val in enumerate(unique_vals):
            weighted_scores[k] = np.dot(self.model_weights_, (predictions == val))
        synthetic_prot = unique_vals[np.argmax(weighted_scores, axis=0)]

        # Create masked copy
        X_masked = X.copy()
        X_masked[self.sens_attr] = synthetic_prot

        return X_masked

    def predict(self, X):
        """
        Predict class labels with masked sensitive attributes.

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix.

        Returns
        -------
        y_pred : ndarray
            Predicted class labels.
        """
        check_is_fitted(self, ["estimator_", "extrapolation_models_"])
        X_masked = self._synthesize_sensitive_attr(X)
        return self.estimator_.predict(X_masked)

    def predict_proba(self, X):
        """
        Predict class probabilities with masked sensitive attributes.

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
        check_is_fitted(self, ["estimator_", "extrapolation_models_"])
        X_masked = self._synthesize_sensitive_attr(X)
        return self.estimator_.predict_proba(X_masked)

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
        check_is_fitted(self, ["estimator_", "extrapolation_models_"])
        X_masked = self._synthesize_sensitive_attr(X)
        return self.estimator_.score(X_masked, y, sample_weight=sample_weight)
