"""
Pipeline building and cross-validation runner for the Experiment class.

Mirrors the workflow in ``examples/comparison_dev.ipynb`` but parameterised
so it can be driven by registry look-ups.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    train_test_split,
)
from imblearn.pipeline import Pipeline as ImbPipeline

from ._registry import METHOD_REGISTRY, _import_object


def _resolve_strat_label(stratify, y_arr, X, sens_col):
    """Return the array used as the stratification label, or None.

    Accepts: None, "none", "y", "sens", "sens_attr", "both".
    """
    if stratify is None or stratify == "none":
        return None
    if stratify == "y":
        return y_arr
    if stratify in ("sens", "sens_attr"):
        return np.asarray(X[sens_col].values)
    if stratify == "both":
        sens = np.asarray(X[sens_col].values)
        return (
            pd.Series(y_arr).astype(str).values
            + "_"
            + pd.Series(sens).astype(str).values
        )
    raise ValueError(
        f"Unknown stratify value: {stratify!r}. "
        "Expected one of: None, 'none', 'y', 'sens', 'sens_attr', 'both'."
    )


def build_pipeline(method_name, clf, X, sens_attr, method_params=None):
    """Build an imblearn ``Pipeline`` for *(method, classifier)*.

    Parameters
    ----------
    method_name : str
        Key in ``METHOD_REGISTRY`` (e.g. ``"FairSmote"``).
    clf : estimator
        Scikit-learn compatible classifier instance.
    X : DataFrame
        Training features — used only to resolve ``repair_columns`` for
        ``DisparateImpactRemover``.
    sens_attr : str
        Sensitive-attribute column name (auto-injected into method kwargs).
    method_params : dict or None
        Extra keyword arguments forwarded to the method constructor.
        These override registry defaults.

    Returns
    -------
    ImbPipeline
    """
    info = METHOD_REGISTRY[method_name]
    category = info["category"]
    clf = clone(clf)

    if category == "baseline":
        return ImbPipeline([("clf", clf)])

    # Merge: registry defaults → user overrides → sens_attr
    kw = {**info["defaults"]}
    if method_params:
        kw.update(method_params)
    kw["sens_attr"] = sens_attr

    MethodClass = _import_object(info["path"])

    if category == "sampler":
        return ImbPipeline([("method", MethodClass(**kw)), ("clf", clf)])

    if category == "repair":
        # DisparateImpactRemover needs repair_columns
        if method_name == "DisparateImpactRemover" and "repair_columns" not in kw:
            numeric_cols = [
                c for c in X.select_dtypes(include=["number"]).columns
                if c != sens_attr
            ]
            kw["repair_columns"] = numeric_cols
        return ImbPipeline([("method", MethodClass(**kw)), ("clf", clf)])

    if category == "meta":
        return ImbPipeline([("clf", MethodClass(estimator=clf, **kw))])

    raise ValueError(f"Unknown category '{category}' for method '{method_name}'")


def _build_splits(n_splits, n_repeats, strat_label, y_arr, random_state):
    """Build the list of (train_idx, test_idx) tuples for the CV procedure.

    Falls back to ``y``-stratification with a warning if the requested
    stratification label has a stratum smaller than ``n_splits``.
    """
    n = len(y_arr)

    def _too_small(label):
        if label is None:
            return False
        _, counts = np.unique(label, return_counts=True)
        # For n_splits == 1 (single hold-out), need at least 2 per stratum
        # so train and test each get one. For K-fold, need at least n_splits.
        threshold = max(n_splits, 2)
        return counts.min() < threshold

    if strat_label is not None and _too_small(strat_label):
        if not np.array_equal(strat_label, y_arr):
            warnings.warn(
                "Stratification failed: a stratum has fewer members than "
                f"n_splits={n_splits}; falling back to stratify='y'.",
                stacklevel=3,
            )
            strat_label = y_arr
            if _too_small(strat_label):
                strat_label = None  # last-resort: no stratification

    def _build(label):
        if n_splits >= 2:
            if label is None:
                splitter = RepeatedKFold(
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    random_state=random_state,
                )
                return list(splitter.split(np.arange(n)))
            splitter = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            return list(splitter.split(np.arange(n), label))
        # n_splits == 1: manual loop of single splits with seed offset
        out = []
        indices = np.arange(n)
        for i in range(n_repeats):
            train_idx, test_idx = train_test_split(
                indices,
                test_size=0.2,
                stratify=label,
                random_state=random_state + i,
            )
            out.append((train_idx, test_idx))
        return out

    return _build(strat_label)


def run_cv(
    pipeline,
    X,
    y,
    sens_col,
    metrics,
    metric_types,
    n_splits=5,
    random_state=42,
    store_predictions=False,
    include_std=False,
    return_model=False,
    stratify="y",
    n_repeats=1,
    priv_group=1,
):
    """Run cross-validation and compute metrics.

    Parameters
    ----------
    pipeline : Pipeline
        As returned by :func:`build_pipeline`.
    X : DataFrame
        Feature matrix (includes the sensitive-attribute column).
    y : array-like
        Binary target.
    sens_col : str
        Sensitive-attribute column name.
    metrics : dict
        ``{name: callable}`` — each callable is a metric function.
    metric_types : dict
        ``{name: "performance"|"fairness"}`` — determines the call signature.
    n_splits : int
        Number of CV folds.  ``1`` triggers a single train/test split.
    random_state : int
        Random seed.
    store_predictions : bool
        If *True*, out-of-fold predictions are collected and returned.
    return_model : bool
        If *True*, the fitted pipeline from the last fold is returned as a
        third element.
    stratify : {None, "none", "y", "sens", "sens_attr", "both"}, default "y"
        Label used for stratified splits. ``None``/``"none"`` disables
        stratification, ``"y"`` stratifies on the target, ``"sens"`` (alias
        ``"sens_attr"``) stratifies on the sensitive attribute, and
        ``"both"`` stratifies on the joint ``(y, sens_attr)`` label.
    n_repeats : int, default 1
        Number of times to repeat the full splitting procedure with
        different seeds. Total fold count is ``n_splits * n_repeats``.
    priv_group : int or str, default 1
        Value of the sensitive attribute treated as the privileged group.
        The attribute is binarised as ``(sens == priv_group)`` before
        fairness metrics are computed (and before predictions are stored),
        so multi-valued attributes are evaluated privileged-vs-rest,
        consistently with the auditors. The default leaves already-binary
        0/1 columns unchanged.

    Returns
    -------
    result : dict
        ``{"{metric}": float, ...}`` plus ``{"{metric}_std": float}``
        when *include_std* is True.
    predictions : dict or None
        ``{"y_true": array, "y_pred": array, "sens_attr": array}`` when
        *store_predictions* is True, else None.
    last_model : Pipeline or None
        The fitted pipeline from the last CV fold when *return_model* is True,
        else None.
    """
    X = X.reset_index(drop=True)
    y_arr = np.asarray(y)

    fold_metrics = {name: [] for name in metrics}

    # Collect out-of-fold predictions
    oof_y_true, oof_y_pred, oof_sens = [], [], []
    last_pipe = None

    strat_label = _resolve_strat_label(stratify, y_arr, X, sens_col)
    splits = _build_splits(n_splits, n_repeats, strat_label, y_arr, random_state)

    for train_idx, test_idx in splits:
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]
        # Binarise privileged-vs-rest so multi-valued sensitive attributes
        # are handled consistently with the auditors (no-op for 0/1 columns
        # with the default priv_group=1).
        sens_test = (X_test[sens_col].values == priv_group).astype(int)

        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        if return_model:
            last_pipe = pipe

        for name, fn in metrics.items():
            if metric_types[name] == "fairness":
                fold_metrics[name].append(fn(y_test, y_pred, sens_test))
            else:
                fold_metrics[name].append(fn(y_test, y_pred))

        if store_predictions:
            oof_y_true.append(y_test)
            oof_y_pred.append(y_pred)
            oof_sens.append(sens_test)

    result = {}
    n_folds_total = len(splits)
    for m, vals in fold_metrics.items():
        result[m] = float(np.nanmean(vals)) if len(vals) else float("nan")
        if include_std:
            result[f"{m}_std"] = (
                float(np.nanstd(vals)) if n_folds_total >= 2 else 0.0
            )

    predictions = None
    if store_predictions:
        predictions = {
            "y_true": np.concatenate(oof_y_true),
            "y_pred": np.concatenate(oof_y_pred),
            "sens_attr": np.concatenate(oof_sens),
            "folds": [
                {"y_true": yt, "y_pred": yp, "sens_attr": sa}
                for yt, yp, sa in zip(oof_y_true, oof_y_pred, oof_sens)
            ],
        }

    return result, predictions, last_pipe
