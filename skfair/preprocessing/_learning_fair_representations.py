"""
Learning Fair Representations (Zemel et al., 2013)

Finds a latent representation that encodes data well but obfuscates
information about sensitive attributes.

Reference:
    R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,
    "Learning Fair Representations."
    International Conference on Machine Learning, 2013.

Based on the AIF360 implementation and
https://github.com/zjelveh/learning-fair-representations
"""

import numpy as np
import scipy.optimize as optim
from scipy.spatial.distance import cdist
from scipy.special import softmax

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state


def _get_xhat_y_hat(prototypes, w, x):
    """Compute soft membership, reconstructed features, and predicted labels.

    Parameters
    ----------
    prototypes : ndarray of shape (k, d)
        Prototype vectors in feature space.
    w : ndarray of shape (k,)
        Prototype-to-label weights.
    x : ndarray of shape (n, d)
        Input feature matrix.

    Returns
    -------
    M : ndarray of shape (n, k)
        Soft membership matrix (softmax over negative distances).
    x_hat : ndarray of shape (n, d)
        Reconstructed features.
    y_hat : ndarray of shape (n, 1)
        Predicted label probabilities, clipped to (eps, 1-eps).
    """
    M = softmax(-cdist(x, prototypes), axis=1)
    x_hat = np.matmul(M, prototypes)
    y_hat = np.clip(
        np.matmul(M, w.reshape((-1, 1))),
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
    )
    return M, x_hat, y_hat


def _lfr_optim_objective(
    parameters,
    x_unprivileged,
    x_privileged,
    y_unprivileged,
    y_privileged,
    k,
    Ax,
    Ay,
    Az,
):
    """Optimization objective for LFR: L = Ax*L_x + Ay*L_y + Az*L_z.

    Parameters
    ----------
    parameters : ndarray
        Flattened array: first k values are prototype-to-label weights w,
        remaining k*d values are prototype coordinates.
    x_unprivileged : ndarray of shape (n0, d)
    x_privileged : ndarray of shape (n1, d)
    y_unprivileged : ndarray of shape (n0,)
    y_privileged : ndarray of shape (n1,)
    k : int
        Number of prototypes.
    Ax, Ay, Az : float
        Loss term weights.

    Returns
    -------
    total_loss : float
    """
    features_dim = x_unprivileged.shape[1]

    w = parameters[:k]
    prototypes = parameters[k:].reshape((k, features_dim))

    M_unpriv, x_hat_unpriv, y_hat_unpriv = _get_xhat_y_hat(
        prototypes, w, x_unprivileged
    )
    M_priv, x_hat_priv, y_hat_priv = _get_xhat_y_hat(
        prototypes, w, x_privileged
    )

    y_hat = np.concatenate([y_hat_unpriv, y_hat_priv], axis=0)
    y = np.concatenate(
        [y_unprivileged.reshape((-1, 1)), y_privileged.reshape((-1, 1))],
        axis=0,
    )

    # Reconstruction loss
    L_x = np.mean((x_hat_unpriv - x_unprivileged) ** 2) + np.mean(
        (x_hat_priv - x_privileged) ** 2
    )
    # Fairness loss: difference in average prototype memberships
    L_z = np.mean(
        abs(np.mean(M_unpriv, axis=0) - np.mean(M_priv, axis=0))
    )
    # Prediction loss (cross-entropy)
    L_y = -np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))

    return Ax * L_x + Ay * L_y + Az * L_z


class LearningFairRepresentations(BaseEstimator, TransformerMixin):
    """Learning Fair Representations (Zemel et al., 2013).

    Learns a set of intermediate prototypes that simultaneously encode
    the data faithfully (low reconstruction error) while removing
    information about a sensitive attribute (statistical parity of
    prototype membership across groups).

    The objective minimised during ``fit`` is::

        L = Ax * L_x  +  Ay * L_y  +  Az * L_z

    where *L_x* is reconstruction error, *L_y* is label-prediction
    cross-entropy, and *L_z* penalises differences in average prototype
    membership between the privileged and unprivileged groups.

    ``transform`` replaces every numeric feature column with its
    prototype-based reconstruction, preserving the sensitive-attribute
    column unchanged.

    Parameters
    ----------
    sens_attr : str
        Column name of the sensitive attribute in X.

    priv_group : int or str
        Value identifying the privileged group in
        ``X[sens_attr]``.

    k : int, default=5
        Number of prototypes.

    Ax : float, default=0.01
        Weight of the reconstruction loss term.

    Ay : float, default=1.0
        Weight of the label-prediction loss term.

    Az : float, default=50.0
        Weight of the fairness loss term.

    maxiter : int, default=5000
        Maximum iterations for L-BFGS-B.

    maxfun : int, default=5000
        Maximum function evaluations for L-BFGS-B.

    random_state : int or None, default=None
        Seed for reproducibility.

    verbose : int, default=0
        Verbosity level.  If > 0, prints the optimization result.

    Attributes
    ----------
    w_ : ndarray of shape (k,)
        Learned prototype-to-label weights.

    prototypes_ : ndarray of shape (k, features_dim_)
        Learned prototype vectors.

    features_dim_ : int
        Number of numeric feature columns seen during ``fit``.

    feature_columns_ : list of str
        Numeric column names used for the representation (excludes
        ``sens_attr``).

    References
    ----------
    .. [1] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,
       "Learning Fair Representations", ICML 2013.
    """

    def __init__(
        self,
        sens_attr,
        priv_group,
        k=5,
        Ax=0.01,
        Ay=1.0,
        Az=50.0,
        maxiter=5000,
        maxfun=5000,
        random_state=None,
        verbose=0,
    ):
        self.sens_attr = sens_attr
        self.priv_group = priv_group
        self.k = k
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        self.maxiter = maxiter
        self.maxfun = maxfun
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """Learn fair prototypes from the data.

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix.  Must contain ``sens_attr`` as a
            column.  All other numeric columns are used as features.

        y : array-like of shape (n_samples,)
            Binary labels (0/1).

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if self.sens_attr not in X.columns:
            raise ValueError(
                f"Protected attribute '{self.sens_attr}' "
                "not found in X."
            )

        rng = check_random_state(self.random_state)
        y = np.asarray(y, dtype=float)

        # Identify numeric feature columns (exclude sensitive attribute)
        self.feature_columns_ = [
            c
            for c in X.select_dtypes(include=[np.number]).columns
            if c != self.sens_attr
        ]
        self.features_dim_ = len(self.feature_columns_)

        features = X[self.feature_columns_].values
        prot = X[self.sens_attr].values

        priv_mask = prot == self.priv_group
        unpriv_mask = ~priv_mask

        x_priv = features[priv_mask]
        x_unpriv = features[unpriv_mask]
        y_priv = y[priv_mask]
        y_unpriv = y[unpriv_mask]

        # Initialise: w in [0,1], prototypes unbounded
        n_params = self.k + self.features_dim_ * self.k
        parameters_init = rng.uniform(size=n_params)
        bounds = [(0, 1)] * self.k + [(None, None)] * (
            self.features_dim_ * self.k
        )

        learned, _, info = optim.fmin_l_bfgs_b(
            _lfr_optim_objective,
            x0=parameters_init,
            epsilon=1e-5,
            args=(
                x_unpriv,
                x_priv,
                y_unpriv,
                y_priv,
                self.k,
                self.Ax,
                self.Ay,
                self.Az,
            ),
            bounds=bounds,
            approx_grad=True,
            maxfun=self.maxfun,
            maxiter=self.maxiter,
            disp=self.verbose,
        )

        self.w_ = learned[: self.k]
        self.prototypes_ = learned[self.k :].reshape(
            (self.k, self.features_dim_)
        )

        return self

    def transform(self, X):
        """Replace numeric features with fair representations.

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix (same schema as ``fit``).

        Returns
        -------
        X_fair : pandas DataFrame
            Copy of X with numeric feature columns replaced by their
            prototype-based reconstructions.  The sensitive-attribute
            column is preserved unchanged.
        """
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        features = X[self.feature_columns_].values
        _, x_hat, _ = _get_xhat_y_hat(self.prototypes_, self.w_, features)

        X_fair = X.copy()
        X_fair[self.feature_columns_] = x_hat
        return X_fair
