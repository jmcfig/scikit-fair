"""
Optimized Pre-Processing for Discrimination Prevention (Calmon et al., 2017)

Learns a probabilistic transformation of features and labels to reduce
discrimination while preserving data utility and limiting individual
distortion, via convex optimization.

Reference
---------
F. P. Calmon, D. Wei, B. Vinzamuri, K. Natesan Ramamurthy, and
K. R. Varshney, "Optimized Pre-Processing for Discrimination Prevention,"
Advances in Neural Information Processing Systems (NeurIPS), 2017.

Adapted from the AIF360 implementation to follow scikit-fair conventions.
"""

import itertools

import numpy as np
import pandas as pd

from ._base import BaseFairSampler


class OptimizedPreprocessing(BaseFairSampler):
    """
    Optimized Pre-Processing for Discrimination Prevention.

    Learns a randomised mapping P(X', Y' | D, X, Y) that transforms features
    and labels to reduce statistical parity disparity while controlling
    individual distortion, solved via convex optimisation (CVXPY).

    **Important**: This algorithm operates on discrete/categorical features
    only.  Continuous features must be discretised before use.

    Parameters
    ----------
    sens_attr : str
        Column name of the sensitive attribute in X.

    features_to_transform : list of str
        Categorical feature columns to include in the optimisation.

    distortion_fun : callable
        Cost function ``(old_dict, new_dict) -> float`` where each dict
        maps feature names to values (plus ``'label'`` for the outcome).

    epsilon : float, default=0.05
        Maximum allowed disparity |P(Y'=y|D=d1) - P(Y'=y|D=d2)|.

    clist : list of float, optional
        Distortion thresholds for the excess-distortion constraint (Eq. 5).
        Defaults to ``[0.99, 1.99, 2.99]``.

    dlist : list of float, optional
        Maximum probability of exceeding each threshold in *clist*.
        Defaults to ``[0.1, 0.05, 0.01]``.  Must have same length as *clist*.

    random_state : int or None, default=None
        Seed for the random number generator used in the randomised mapping.

    Attributes
    ----------
    mapping_ : dict
        Learned conditional distribution P(X', Y' | D, X, Y).

    classes_ : tuple
        (neg_label, pos_label) observed during fitting.

    Example
    -------
    >>> from skfair.preprocessing import OptimizedPreprocessing
    >>> def distortion(old, new):
    ...     cost = 0.0
    ...     for k in old:
    ...         if k != 'label' and old[k] != new[k]:
    ...             cost += 1.0
    ...     if old['label'] != new['label']:
    ...         cost += 2.0
    ...     return cost
    >>> op = OptimizedPreprocessing(
    ...     sens_attr='group',
    ...     features_to_transform=['age_cat', 'education'],
    ...     distortion_fun=distortion,
    ...     epsilon=0.05,
    ... )
    """

    _sampling_type = "clean-sampling"

    def __init__(
        self,
        sens_attr=None,
        features_to_transform=None,
        distortion_fun=None,
        epsilon=0.05,
        clist=None,
        dlist=None,
        random_state=None,
    ):
        super().__init__(sens_attr, random_state)
        self.features_to_transform = features_to_transform
        self.distortion_fun = distortion_fun
        self.epsilon = epsilon
        self.clist = clist
        self.dlist = dlist

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _check_X_y(self, X, y):
        """Override to add sens_attr None check before base validation."""
        if self.sens_attr is None:
            raise ValueError("sens_attr parameter must be specified.")
        return super()._check_X_y(X, y)

    # ------------------------------------------------------------------
    # Core resampling logic
    # ------------------------------------------------------------------

    def _fit_resample(self, X, y):
        """
        Fit the optimisation model and apply the mapping P(X', Y' | D, X, Y).

        Parameters
        ----------
        X : pandas DataFrame
            Must contain ``sens_attr`` and all ``features_to_transform``
            columns.  All values must be discrete / categorical.

        y : array-like of shape (n_samples,)
            Binary labels.

        Returns
        -------
        X_resampled : pandas DataFrame
            Transformed features (same shape as X).

        y_resampled : numpy ndarray
            Transformed labels.
        """
        y_series = self._align_y(X, y)
        self._validate_op_params(X, y_series)

        self._build_joint_distribution(X, y_series)
        self._build_distortion_matrix()
        self._solve_optimization()

        # persistent RNG
        self._rng = np.random.RandomState(self.random_state)

        return self._apply_mapping(X, y_series)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_y(X, y):
        """Return *y* as a Series aligned on X's index."""
        if isinstance(y, pd.Series):
            s = y.copy()
            s.index = X.index
            return s
        return pd.Series(y, index=X.index)

    def _validate_op_params(self, X, y_series):
        """Validate inputs."""
        missing = set(self.features_to_transform) - set(X.columns)
        if missing:
            raise ValueError(
                f"features_to_transform columns not found in X: {missing}"
            )

        if not callable(self.distortion_fun):
            raise TypeError("distortion_fun must be callable.")

        unique_labels = pd.unique(y_series)
        if len(unique_labels) != 2:
            raise ValueError(
                f"Binary labels required, got {len(unique_labels)} unique "
                f"values: {unique_labels}"
            )

        # Determine and store class labels (sorted so 0 < 1 by default)
        sorted_labels = sorted(unique_labels)
        self.classes_ = (sorted_labels[0], sorted_labels[1])

        # Defaults for clist / dlist
        clist = self.clist if self.clist is not None else [0.99, 1.99, 2.99]
        dlist = self.dlist if self.dlist is not None else [0.1, 0.05, 0.01]
        if len(clist) != len(dlist):
            raise ValueError(
                f"clist and dlist must have the same length, "
                f"got {len(clist)} and {len(dlist)}."
            )
        self._clist = clist
        self._dlist = dlist

    # ------------------------------------------------------------------

    def _build_joint_distribution(self, X, y_series):
        """Compute empirical P(D, X, Y) and catalogue all combos (full support)."""
        d_col = self.sens_attr
        feat_cols = list(self.features_to_transform)

        # Unique values per dimension
        self._d_values = sorted(X[d_col].unique())
        self._y_values = sorted(y_series.unique())
        self._feat_domains = {
            c: sorted(X[c].unique()) for c in feat_cols
        }

        # All possible x-tuples (cartesian product of feature domains)
        domain_lists = [self._feat_domains[c] for c in feat_cols]
        self._x_tuples = list(itertools.product(*domain_lists))

        # Map every (d, x, y) to an index and compute its probability
        n = len(X)
        self._dxy_keys = []  # list of (d, x_tuple, y)
        self._dxy_prob = {}  # key -> probability

        for d in self._d_values:
            for x_tup in self._x_tuples:
                for yv in self._y_values:
                    key = (d, x_tup, yv)

                    # Count occurrences
                    mask = (X[d_col] == d) & (y_series == yv)
                    for col, val in zip(feat_cols, x_tup):
                        mask = mask & (X[col] == val)
                    count = mask.sum()

                    # IMPORTANT: include all combinations, even if count == 0
                    self._dxy_keys.append(key)
                    self._dxy_prob[key] = count / n

        # All possible (x', y') outcomes
        self._xy_outcomes = []
        for x_tup in self._x_tuples:
            for yv in self._y_values:
                self._xy_outcomes.append((x_tup, yv))

        # Marginal P(D=d)
        self._p_d = {}
        for d in self._d_values:
            self._p_d[d] = (X[d_col] == d).sum() / n


    # ------------------------------------------------------------------

    def _build_distortion_matrix(self):
        """Build cost matrix D[i, j] for all (d,x,y) -> (x',y') pairs."""
        feat_cols = list(self.features_to_transform)
        n_rows = len(self._dxy_keys)
        n_cols = len(self._xy_outcomes)
        self._distortion_matrix = np.zeros((n_rows, n_cols))

        for i, (_, x_tup, y) in enumerate(self._dxy_keys):
            old_dict = {c: v for c, v in zip(feat_cols, x_tup)}
            old_dict["label"] = y

            for j, (x_tup_new, y_new) in enumerate(self._xy_outcomes):
                new_dict = {c: v for c, v in zip(feat_cols, x_tup_new)}
                new_dict["label"] = y_new
                self._distortion_matrix[i, j] = self.distortion_fun(
                    old_dict, new_dict
                )

    # ------------------------------------------------------------------

    def _solve_optimization(self):
        """Set up and solve the CVXPY convex programme."""
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError(
                "OptimizedPreprocessing requires cvxpy. "
                "Install it with: pip install cvxpy"
            )

        n_rows = len(self._dxy_keys)
        n_cols = len(self._xy_outcomes)

        # Decision variable: conditional mapping P(x',y' | d,x,y)
        P = cp.Variable((n_rows, n_cols), nonneg=True)

        constraints = []

        # 1. Each row sums to 1 (valid conditional distribution)
        for i in range(n_rows):
            constraints.append(cp.sum(P[i, :]) == 1)

        # 2. Fairness constraints: |P'(Y'=y | D=d) - P'(Y'=y | D=d')| <= eps
        #    P'(Y'=y | D=d) = sum over (d,x,y_orig) with this d of
        #        P(d,x,y_orig) * sum_j [P[i,j] where outcome j has Y'=y]
        #    ... divided by P(D=d)
        for y_val in self._y_values:
            # Columns whose outcome has this y value
            y_col_mask = np.array(
                [1.0 if out_y == y_val else 0.0
                 for (_, out_y) in self._xy_outcomes]
            )

            # P'(Y'=y_val | D=d) for each d
            p_y_given_d = {}
            for d in self._d_values:
                expr = 0.0
                for i, (dk, _, _) in enumerate(self._dxy_keys):
                    if dk == d:
                        prob_i = self._dxy_prob[self._dxy_keys[i]]
                        expr = expr + prob_i * (P[i, :] @ y_col_mask)
                p_d = self._p_d[d]
                if p_d > 0:
                    p_y_given_d[d] = expr / p_d

            # Pairwise fairness
            d_list = list(p_y_given_d.keys())
            for a in range(len(d_list)):
                for b in range(a + 1, len(d_list)):
                    diff = p_y_given_d[d_list[a]] - p_y_given_d[d_list[b]]
                    constraints.append(diff <= self.epsilon)
                    constraints.append(diff >= -self.epsilon)

        # 3. Excess-distortion constraints (Eq. 5)
        #    For each threshold c_k:
        #      sum_{i,j where D[i,j] > c_k} P(d,x,y) * P[i,j] <= d_k
        for c_thresh, d_thresh in zip(self._clist, self._dlist):
            indicator = (self._distortion_matrix > c_thresh).astype(float)
            expr = 0.0
            for i in range(n_rows):
                prob_i = self._dxy_prob[self._dxy_keys[i]]
                expr = expr + prob_i * (P[i, :] @ indicator[i, :])
            constraints.append(expr <= d_thresh)

        # Objective: minimise L1 distance between P(X,Y) and P'(X,Y)
        # P'(x',y') = sum_{d,x,y} P(d,x,y) * P[i,j]
        # P(x,y)    = sum_d P(d,x,y)
        p_xy_original = np.zeros(n_cols)
        for j, (x_tup_out, y_out) in enumerate(self._xy_outcomes):
            for i, key in enumerate(self._dxy_keys):
                _, x_k, y_k = key
                if x_k == x_tup_out and y_k == y_out:
                    p_xy_original[j] += self._dxy_prob[key]

        p_xy_new_expr = []
        for j in range(n_cols):
            expr_j = 0.0
            for i in range(n_rows):
                prob_i = self._dxy_prob[self._dxy_keys[i]]
                expr_j = expr_j + prob_i * P[i, j]
            p_xy_new_expr.append(expr_j)

        # L1 distance = sum |p_original_j - p_new_j|
        objective_terms = []
        for j in range(n_cols):
            objective_terms.append(cp.abs(p_xy_original[j] - p_xy_new_expr[j]))

        objective = cp.Minimize(cp.sum(objective_terms))

        problem = cp.Problem(objective, constraints)

        # Try solvers in order of preference
        for solver, kwargs in [
            (cp.CLARABEL, {}),
            (cp.SCS, {"max_iters": 10000}),
        ]:
            try:
                problem.solve(solver=solver, verbose=False, **kwargs)
                if P.value is not None:
                    break
            except cp.error.SolverError:
                continue

        if problem.status == "infeasible":
            raise RuntimeError(
                "Optimisation problem is infeasible. Try relaxing epsilon, "
                "clist/dlist constraints, or providing more training data."
            )

        # Store the solved mapping
        P_val = P.value
        if P_val is None:
            raise RuntimeError(
                "Solver did not return a solution. Try relaxing constraints."
            )

        # Clip small numerical artefacts
        P_val = np.clip(P_val, 0, None)
        # Re-normalise rows
        row_sums = P_val.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        P_val = P_val / row_sums

        # Build lookup: (d, x_tuple, y) -> array of probs over outcomes
        self.mapping_ = {}
        for i, key in enumerate(self._dxy_keys):
            self.mapping_[key] = P_val[i, :]

    # ------------------------------------------------------------------

    def _apply_mapping(self, X, y_series):
        """
        Apply the randomised joint mapping to training data.

        Parameters
        ----------
        X : DataFrame
        y_series : Series

        Returns
        -------
        X_out : DataFrame
        y_out : ndarray
        """
        rng = self._rng
        feat_cols = list(self.features_to_transform)

        X_out = X.copy()
        y_out = y_series.copy()

        for idx in X.index:
            d = X.at[idx, self.sens_attr]
            x_tup = tuple(X.at[idx, c] for c in feat_cols)
            yv = y_series.at[idx]

            key = (d, x_tup, yv)
            if key in self.mapping_:
                probs = self.mapping_[key]
                # Draw outcome
                j = rng.choice(len(self._xy_outcomes), p=probs)
                x_new, y_new = self._xy_outcomes[j]
                for c, v in zip(feat_cols, x_new):
                    X_out.at[idx, c] = v
                y_out.at[idx] = y_new
            # else: identity (unseen combo, leave unchanged)

        return X_out, y_out.values
