"""
Pipeline building and cross-validation runner for the Experiment class.

Mirrors the workflow in ``examples/comparison_dev.ipynb`` but parameterised
so it can be driven by registry look-ups.
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline

from ._registry import METHOD_REGISTRY, _import_object


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
        ``GeometricFairnessRepair``.
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
        # GeometricFairnessRepair needs repair_columns
        if method_name == "GeometricFairnessRepair" and "repair_columns" not in kw:
            numeric_cols = [
                c for c in X.select_dtypes(include=["number"]).columns
                if c != sens_attr
            ]
            kw["repair_columns"] = numeric_cols
        return ImbPipeline([("method", MethodClass(**kw)), ("clf", clf)])

    if category == "meta":
        return ImbPipeline([("clf", MethodClass(estimator=clf, **kw))])

    raise ValueError(f"Unknown category '{category}' for method '{method_name}'")


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
):
    """Run stratified cross-validation and compute metrics.

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

    Returns
    -------
    result : dict
        ``{"{metric}_mean": float, "{metric}_std": float, ...}``
    predictions : dict or None
        ``{"y_true": array, "y_pred": array, "sens_attr": array}`` when
        *store_predictions* is True, else None.
    """
    X = X.reset_index(drop=True)
    y_arr = np.asarray(y)

    fold_metrics = {name: [] for name in metrics}

    # Collect out-of-fold predictions
    oof_y_true, oof_y_pred, oof_sens = [], [], []

    if n_splits >= 2:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = list(skf.split(X, y_arr))
    else:
        # Single train/test split
        indices = np.arange(len(y_arr))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=y_arr,
            random_state=random_state,
        )
        splits = [(train_idx, test_idx)]

    for train_idx, test_idx in splits:
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]
        sens_test = X_test[sens_col].values

        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

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
    for m, vals in fold_metrics.items():
        result[f"{m}_mean"] = float(np.mean(vals))
        result[f"{m}_std"] = float(np.std(vals)) if n_splits >= 2 else 0.0

    predictions = None
    if store_predictions:
        predictions = {
            "y_true": np.concatenate(oof_y_true),
            "y_pred": np.concatenate(oof_y_pred),
            "sens_attr": np.concatenate(oof_sens),
        }

    return result, predictions
