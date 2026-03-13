"""
Experiment class — orchestrates dataset × method × classifier comparisons.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ._registry import (
    DATASET_REGISTRY,
    METHOD_REGISTRY,
    METRIC_REGISTRY,
    _import_object,
)
from ._config_parser import parse_experiment_config
from ._runner import build_pipeline, run_cv


class Experiment:
    """Run a fairness-method comparison experiment.

    Parameters
    ----------
    datasets : list of str or dict, optional
        Each element is either a string (key of ``DATASET_REGISTRY``) or a
        dict with keys ``"name"``, ``"data"``, ``"sens_attr"`` and optionally
        ``"priv_group"`` (default 1).  Example::

            datasets=[
                "ricci",
                {"name": "my_data", "data": (X, y),
                 "sens_attr": "gender", "priv_group": 1},
            ]

        Default: ``["adult"]``.
    methods : list of str, optional
        Method names (keys of ``METHOD_REGISTRY``).  Default: all methods.
    classifiers : dict or list, optional
        Either ``{"name": estimator_instance}`` or a list of dotted import
        paths (e.g. ``["sklearn.svm.SVC"]``).
        Default: ``{"LogReg": LogisticRegression(...)}``.
    metrics : list of str, optional
        Metric names (keys of ``METRIC_REGISTRY``).  Default: all 6 metrics.
    n_splits : int
        Number of CV folds (``1`` for a single train/test split).
    random_state : int
        Random seed.
    dataset_config : dict, optional
        Per-dataset overrides, e.g.
        ``{"adult": {"sens_attr": "race", "priv_group": 1}}``.
    method_config : dict, optional
        Per-method param overrides, e.g.
        ``{"FairSmote": {"random_state": 0}}``.
    std : bool
        If *True*, include ``{metric}_std`` columns in the results DataFrame.
    audit_bias : bool
        If *True*, create a ``BiasAuditor`` per dataset after loading.
    audit_fairness : bool
        If *True*, store out-of-fold predictions so that
        :meth:`audit_fairness` can build a ``FairnessAuditor`` later.
    config : str, optional
        Path to (or raw string of) a YAML configuration.  When provided,
        all other arguments are **ignored** and the config is read from YAML.
    """

    def __init__(
        self,
        datasets=None,
        methods=None,
        classifiers=None,
        metrics=None,
        n_splits=5,
        random_state=42,
        dataset_config=None,
        method_config=None,
        std=False,
        audit_bias=False,
        audit_fairness=False,
        save_results=False,
        save_object=False,
        save_path="experiment",
        config=None,
    ):
        if config is not None:
            self._init_from_config(config)
            return

        # -- datasets --
        self.datasets = self._resolve_datasets(datasets or ["adult"])

        # -- methods --
        self.methods = self._validate_methods(
            methods or list(METHOD_REGISTRY.keys())
        )

        # -- classifiers --
        self.classifiers = self._resolve_classifiers(
            classifiers
            or {
                "LogReg": LogisticRegression(
                    solver="liblinear", max_iter=1000, random_state=42
                )
            }
        )

        # -- metrics --
        self.metrics = self._validate_metrics(
            metrics or list(METRIC_REGISTRY.keys())
        )

        self.n_splits = n_splits
        self.random_state = random_state
        self.dataset_config = dataset_config or {}
        self.method_config = method_config or {}
        self.std = std
        self.audit_bias = audit_bias
        self.audit_fairness = audit_fairness
        self.save_results = save_results
        self.save_object = save_object
        self.save_path = save_path

        # Populated by run()
        self.results_ = None
        self.bias_reports_ = {}
        self._predictions = {}

    @property
    def dataset_names(self):
        """Return list of dataset display names."""
        return [d["name"] for d in self.datasets]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, path):
        """Create an ``Experiment`` from a YAML configuration file/string."""
        return cls(config=path)

    def _init_from_config(self, source):
        """Parse YAML config and set attributes."""
        cfg = parse_experiment_config(source)

        # datasets
        ds_names = [d["name"] for d in cfg["datasets"]] if cfg["datasets"] else ["adult"]
        self.datasets = self._resolve_datasets(ds_names)
        # build dataset_config from config attributes
        self.dataset_config = {}
        for d in cfg["datasets"]:
            name = d["name"]
            overrides = {k: v for k, v in d.items() if k != "name"}
            if overrides:
                self.dataset_config[name] = overrides

        # methods
        m_names = [m["name"] for m in cfg["methods"]] if cfg["methods"] else list(METHOD_REGISTRY.keys())
        self.methods = self._validate_methods(m_names)
        self.method_config = {}
        for m in cfg["methods"]:
            name = m["name"]
            overrides = {k: v for k, v in m.items() if k != "name"}
            if overrides:
                self.method_config[name] = overrides

        # classifiers
        if cfg["classifiers"]:
            self.classifiers = self._resolve_classifiers(cfg["classifiers"])
        else:
            self.classifiers = {
                "LogReg": LogisticRegression(
                    solver="liblinear", max_iter=1000, random_state=42
                )
            }

        # metrics
        if cfg["metrics"]:
            m_names = [m["name"] for m in cfg["metrics"]]
            self.metrics = self._validate_metrics(m_names)
        else:
            self.metrics = list(METRIC_REGISTRY.keys())

        # cv
        self.n_splits = cfg["cv"].get("n_splits", 5)
        self.random_state = cfg["cv"].get("random_state", 42)

        # std
        self.std = False

        # audit
        self.audit_bias = cfg["audit"].get("bias", False)
        self.audit_fairness = cfg["audit"].get("fairness", False)

        # save (defaults off from config; user can override after from_config())
        self.save_results = False
        self.save_object = False
        self.save_path = "experiment"

        self.results_ = None
        self.bias_reports_ = {}
        self._predictions = {}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_datasets(items):
        """Resolve a list of dataset specs (strings or dicts) into uniform dicts.

        Each resolved dict has keys:
        ``name``, ``source`` ("registry" or "user"), ``key``, ``data``,
        ``sens_attr``, ``priv_group``.
        """
        resolved = []
        for item in items:
            if isinstance(item, str):
                key = item.lower()
                if key not in DATASET_REGISTRY:
                    warnings.warn(
                        f"Unknown dataset '{item}', falling back to 'adult'.",
                        stacklevel=2,
                    )
                    key = "adult"
                resolved.append({
                    "name": key,
                    "source": "registry",
                    "key": key,
                    "data": None,
                    "sens_attr": None,
                    "priv_group": None,
                })
            elif isinstance(item, dict):
                missing = {"name", "data", "sens_attr"} - item.keys()
                if missing:
                    raise ValueError(
                        f"Custom dataset dict is missing required keys: "
                        f"{missing}"
                    )
                resolved.append({
                    "name": item["name"],
                    "source": "user",
                    "key": None,
                    "data": item["data"],
                    "sens_attr": item["sens_attr"],
                    "priv_group": item.get("priv_group", 1),
                })
            else:
                raise TypeError(
                    f"Each dataset must be a str or dict, got "
                    f"{type(item).__name__}"
                )
        return resolved

    @staticmethod
    def _validate_methods(names):
        validated = []
        for name in names:
            if name not in METHOD_REGISTRY:
                warnings.warn(
                    f"Unknown method '{name}', falling back to 'FairSmote'.",
                    stacklevel=2,
                )
                name = "FairSmote"
            validated.append(name)
        return validated

    @staticmethod
    def _validate_metrics(names):
        validated = []
        for name in names:
            if name not in METRIC_REGISTRY:
                warnings.warn(
                    f"Unknown metric '{name}', skipping.", stacklevel=2
                )
                continue
            validated.append(name)
        return validated

    @staticmethod
    def _resolve_classifiers(spec):
        """Accept a dict of instances or a list of config-style dicts/strings."""
        if isinstance(spec, dict):
            # Already {"name": instance} — check all values are estimators
            resolved = {}
            for name, obj in spec.items():
                if isinstance(obj, str):
                    # dotted path as value
                    try:
                        cls = _import_object(obj)
                        resolved[name] = cls()
                    except Exception as exc:
                        warnings.warn(
                            f"Cannot import classifier '{obj}': {exc}. Skipping.",
                            stacklevel=2,
                        )
                else:
                    resolved[name] = obj
            return resolved

        if isinstance(spec, list):
            resolved = {}
            for item in spec:
                if isinstance(item, str):
                    # dotted import path
                    path = item
                    clf_name = path.rpartition(".")[2]
                    params = {}
                elif isinstance(item, dict):
                    path = item["path"]
                    clf_name = item.get("name", path.rpartition(".")[2])
                    params = {
                        k: v for k, v in item.items() if k not in ("path", "name")
                    }
                else:
                    continue

                try:
                    cls = _import_object(path)
                    resolved[clf_name] = cls(**params)
                except Exception as exc:
                    warnings.warn(
                        f"Cannot import classifier '{path}': {exc}. Skipping.",
                        stacklevel=2,
                    )
            return resolved

        raise TypeError(
            f"classifiers must be a dict or list, got {type(spec).__name__}"
        )

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def run(self, verbose=True):
        """Execute the experiment.

        Returns
        -------
        pandas.DataFrame
            One row per (dataset, method, classifier) with metric columns.
            Includes ``{metric}_std`` columns when ``std=True``.
        """
        # Resolve metric callables
        metric_fns = {}
        metric_types = {}
        for name in self.metrics:
            info = METRIC_REGISTRY[name]
            metric_fns[name] = _import_object(info["path"])
            metric_types[name] = info["type"]

        rows = []

        for ds_entry in self.datasets:
            if ds_entry["source"] == "registry":
                ds_key = ds_entry["key"]
                ds_info = {**DATASET_REGISTRY[ds_key]}
                # Apply per-dataset overrides
                ds_info.update(self.dataset_config.get(ds_key, {}))

                loader = _import_object(ds_info["loader"])
                X, y = loader()
                sens_attr = ds_info["sens_attr"]
                priv_group = ds_info.get("priv_group", 1)
                ds_display = ds_key.replace("_", " ").title()
            else:
                X, y = ds_entry["data"]
                sens_attr = ds_entry["sens_attr"]
                priv_group = ds_entry["priv_group"]
                ds_display = ds_entry["name"]
                ds_key = ds_entry["name"]

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Dataset: {ds_display}")
                print("=" * 60)

            # Bias audit
            if self.audit_bias:
                from skfair.audit import BiasAuditor

                self.bias_reports_[ds_key] = BiasAuditor(
                    X, y, sens_attr=sens_attr, priv_group=priv_group
                )

            for method_name in self.methods:
                method_params = self.method_config.get(method_name)

                for clf_name, clf in self.classifiers.items():
                    label = f"{method_name:30s} | {clf_name}"
                    try:
                        pipeline = build_pipeline(
                            method_name, clf, X, sens_attr, method_params
                        )
                        cv_result, preds = run_cv(
                            pipeline,
                            X,
                            y,
                            sens_col=sens_attr,
                            metrics=metric_fns,
                            metric_types=metric_types,
                            n_splits=self.n_splits,
                            random_state=self.random_state,
                            store_predictions=self.audit_fairness,
                            include_std=self.std,
                        )
                        row = {
                            "dataset": ds_display,
                            "method": method_name,
                            "classifier": clf_name,
                            **cv_result,
                        }
                        rows.append(row)

                        if preds is not None:
                            self._predictions[
                                (ds_display, method_name, clf_name)
                            ] = preds

                        if verbose:
                            acc = cv_result.get("accuracy", float("nan"))
                            spd = cv_result.get("spd", float("nan"))
                            print(
                                f"  {label}  acc={acc:.3f}  spd={spd:.3f}"
                            )

                    except Exception as exc:
                        if verbose:
                            print(f"  {label}  FAILED: {exc}")
                        row = {
                            "dataset": ds_display,
                            "method": method_name,
                            "classifier": clf_name,
                        }
                        for m in self.metrics:
                            row[m] = float("nan")
                            if self.std:
                                row[f"{m}_std"] = float("nan")
                        rows.append(row)

        self.results_ = pd.DataFrame(rows)

        if self.save_results or self.save_object:
            self.save()

        return self.results_

    # ------------------------------------------------------------------
    # Post-run analysis
    # ------------------------------------------------------------------

    def get_fairness_auditor(self, dataset, method, classifier):
        """Create a ``FairnessAuditor`` from stored out-of-fold predictions.

        Parameters
        ----------
        dataset, method, classifier : str
            Must match values in the results DataFrame.

        Returns
        -------
        skfair.audit.FairnessAuditor
        """
        if not self.audit_fairness:
            raise RuntimeError(
                "Prediction storage was not enabled. "
                "Re-run with audit_fairness=True."
            )
        if self.results_ is None:
            raise RuntimeError("Call .run() before requesting auditors.")

        key = (dataset, method, classifier)
        if key not in self._predictions:
            raise KeyError(
                f"No predictions stored for {key}. "
                "Check that the combination exists and did not fail."
            )

        preds = self._predictions[key]
        from skfair.audit import FairnessAuditor

        return FairnessAuditor(
            y_true=preds["y_true"],
            y_pred=preds["y_pred"],
            sens_attr=preds["sens_attr"],
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(self, path=None, results=None, object=None):
        """Save experiment outputs.

        Parameters
        ----------
        path : str, optional
            Base path (without extension). Defaults to ``self.save_path``.
        results : bool, optional
            Write results DataFrame to ``{path}.csv``.
            Defaults to ``self.save_results``.
        object : bool, optional
            Pickle full Experiment to ``{path}.pkl``.
            Defaults to ``self.save_object``.
        """
        if self.results_ is None:
            raise RuntimeError("No results to save. Call .run() first.")

        import joblib
        from pathlib import Path

        path = path or self.save_path
        base = Path(path).with_suffix("")
        if results is None:
            results = self.save_results
        if object is None:
            object = self.save_object

        if results:
            self.results_.to_csv(str(base.with_suffix(".csv")), index=False)
        if object:
            joblib.dump(self, str(base.with_suffix(".pkl")))

    @classmethod
    def load(cls, path):
        """Load a previously saved Experiment from a ``.pkl`` file."""
        import joblib

        exp = joblib.load(path)
        if not isinstance(exp, cls):
            raise TypeError(
                f"Loaded object is {type(exp).__name__}, expected Experiment."
            )
        return exp

    def to_report(self):
        """Wrap results in a ``ComparisonReport``."""
        if self.results_ is None:
            raise RuntimeError("No results yet. Call .run() first.")
        from skfair.comparison import ComparisonReport

        return ComparisonReport(self.results_)
