"""
Experiment class — orchestrates dataset × method × classifier comparisons.
"""

import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression

from ._registry import (
    DATASET_REGISTRY,
    DEFAULT_METRICS,
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
    stratify : {None, "none", "y", "sens", "sens_attr", "both"}, default "y"
        Label used for stratified CV splits. ``None``/``"none"`` disables
        stratification, ``"y"`` stratifies on the target, ``"sens"`` (alias
        ``"sens_attr"``) stratifies on the sensitive attribute, and
        ``"both"`` stratifies on the joint ``(y, sens_attr)`` label. If a
        stratum has fewer members than ``n_splits``, falls back to
        ``"y"`` with a warning.
    n_repeats : int, default 1
        Number of times to repeat the full splitting procedure with
        different seeds. Total fold count is ``n_splits * n_repeats``;
        all folds are averaged into a single mean (and ``std`` if
        ``std=True``).
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
    save_models : dict or None
        When not None, fitted pipelines are persisted.  Keys:

        - ``"full_data_retrain"`` (bool, default True): retrain on the full
          dataset before saving.  If False, save the model from the last CV
          fold.
        - ``"models"`` (``"all"`` or list of dicts, default ``"all"``):
          ``"all"`` saves every combination; a list of
          ``{"method": ..., "classifier": ...}`` dicts restricts which
          combinations are saved.
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
        stratify="y",
        n_repeats=1,
        dataset_config=None,
        method_config=None,
        std=False,
        audit_bias=False,
        audit_fairness=False,
        save_results_csv=False,
        save_object_pkl=False,
        save_report_html=False,
        save_path="experiment",
        save_models=None,
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
            metrics or list(DEFAULT_METRICS)
        )

        self.n_splits = n_splits
        self.random_state = random_state
        self.stratify = stratify
        self.n_repeats = n_repeats
        self.dataset_config = dataset_config or {}
        self.method_config = method_config or {}
        self.std = std
        self.audit_bias = audit_bias
        self.audit_fairness = audit_fairness
        self.save_results_csv = save_results_csv
        self.save_object_pkl = save_object_pkl
        self.save_report_html = save_report_html
        self.save_path = save_path
        self.save_models = save_models

        # Populated by run()
        self.results_ = None
        self.bias_reports_ = {}
        self._predictions = {}
        self.models_ = {}

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
            self.metrics = list(DEFAULT_METRICS)

        # cv
        self.n_splits = cfg["cv"].get("n_splits", 5)
        self.random_state = cfg["cv"].get("random_state", 42)
        self.stratify = cfg["cv"].get("stratify", "y")
        self.n_repeats = cfg["cv"].get("n_repeats", 1)

        # std
        self.std = False

        # audit
        self.audit_bias = cfg["audit"].get("bias", False)
        self.audit_fairness = cfg["audit"].get("fairness", False)

        # save
        self.save_results_csv = cfg["save"].get("results_csv", False)
        self.save_object_pkl = cfg["save"].get("object_pkl", False)
        self.save_report_html = cfg["save"].get("report_html", False)
        self.save_path = cfg["save"].get("path", "experiment")

        # save_models
        self.save_models = cfg.get("save_models")

        self.results_ = None
        self.bias_reports_ = {}
        self._predictions = {}
        self.models_ = {}

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

    def _should_save_model(self, method_name, clf_name):
        """Check whether (method_name, clf_name) should be saved."""
        if self.save_models is None:
            return False
        models = self.save_models.get("models", "all")
        if models == "all":
            return True
        if isinstance(models, list):
            return any(
                m.get("method") == method_name and m.get("classifier") == clf_name
                for m in models
            )
        return False

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
                        want_model = self._should_save_model(method_name, clf_name)
                        full_retrain = (
                            self.save_models.get("full_data_retrain", True)
                            if self.save_models
                            else True
                        )
                        cv_result, preds, last_model = run_cv(
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
                            return_model=(want_model and not full_retrain),
                            stratify=self.stratify,
                            n_repeats=self.n_repeats,
                        )
                        row = {
                            "dataset": ds_display,
                            "method": method_name,
                            "classifier": clf_name,
                            **cv_result,
                        }
                        rows.append(row)

                        if want_model:
                            if full_retrain:
                                from sklearn.base import clone as _clone

                                full_pipe = _clone(pipeline)
                                full_pipe.fit(X, y)
                                self.models_[
                                    (ds_display, method_name, clf_name)
                                ] = full_pipe
                            else:
                                self.models_[
                                    (ds_display, method_name, clf_name)
                                ] = last_model

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

        if (
            self.save_results_csv
            or self.save_object_pkl
            or self.save_report_html
            or self.save_models is not None
        ):
            self.save()

        return self.results_

    # ------------------------------------------------------------------
    # Post-run analysis
    # ------------------------------------------------------------------

    def get_fairness_auditor(self, dataset, method, classifier,
                             aggregate=False):
        """Create a ``FairnessAuditor`` from stored out-of-fold predictions.

        Parameters
        ----------
        dataset, method, classifier : str
            Must match values in the results DataFrame.
        aggregate : bool, default=False
            If *True*, all out-of-fold predictions are concatenated into a
            single ``FairnessAuditor`` (one metric computation on all data).
            If *False* (default), fairness metrics are computed per fold and
            averaged, consistent with how the comparison report works.

        Returns
        -------
        skfair.audit.FairnessAuditor
            When ``aggregate=False`` the auditor is built from the first fold
            but its :meth:`fairness_metrics` is monkey-patched to return
            the cross-fold average instead.
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

        if aggregate:
            return FairnessAuditor(
                y_true=preds["y_true"],
                y_pred=preds["y_pred"],
                sens_attr=preds["sens_attr"],
            )

        # Per-fold averaging
        fold_auditors = [
            FairnessAuditor(
                y_true=f["y_true"],
                y_pred=f["y_pred"],
                sens_attr=f["sens_attr"],
            )
            for f in preds["folds"]
        ]

        # Build the primary auditor from concatenated predictions so that
        # tables and plots still work, but override fairness_metrics to
        # return the cross-fold average.
        auditor = FairnessAuditor(
            y_true=preds["y_true"],
            y_pred=preds["y_pred"],
            sens_attr=preds["sens_attr"],
        )

        import pandas as pd

        def _averaged_fairness_metrics():
            dfs = [fa.fairness_metrics() for fa in fold_auditors]
            return pd.concat(dfs).groupby(level=0).mean().reindex(dfs[0].index)

        auditor.fairness_metrics = _averaged_fairness_metrics
        return auditor

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(
        self, path=None, results_csv=None, object_pkl=None, report_html=None,
        models=None,
    ):
        """Save experiment outputs.

        Parameters
        ----------
        path : str, optional
            Base path (without extension). Defaults to ``self.save_path``.
        results_csv : bool, optional
            Write results DataFrame to ``{path}.csv``.
            Defaults to ``self.save_results_csv``.
        object_pkl : bool, optional
            Pickle full Experiment to ``{path}.pkl``.
            Defaults to ``self.save_object_pkl``.
        report_html : bool, optional
            Generate HTML report to ``{path}.html``.
            Defaults to ``self.save_report_html``.
        models : bool, optional
            Save fitted models to ``{path}_models/``.
            Defaults to ``True`` when ``self.save_models`` is not None.
        """
        if self.results_ is None:
            raise RuntimeError("No results to save. Call .run() first.")

        import os
        import joblib
        from pathlib import Path

        path = path or self.save_path
        base = Path(path).with_suffix("")
        if results_csv is None:
            results_csv = self.save_results_csv
        if object_pkl is None:
            object_pkl = self.save_object_pkl
        if report_html is None:
            report_html = self.save_report_html
        if models is None:
            models = self.save_models is not None

        if results_csv:
            self.results_.to_csv(str(base.with_suffix(".csv")), index=False)
        if object_pkl:
            joblib.dump(self, str(base.with_suffix(".pkl")))
        if report_html:
            self.to_report().to_html(str(base.with_suffix(".html")))
        if models and self.models_:
            models_dir = str(base) + "_models"
            os.makedirs(models_dir, exist_ok=True)
            for (ds, method, clf_name), pipe in self.models_.items():
                joblib.dump(
                    pipe,
                    os.path.join(models_dir, f"{ds}_{method}_{clf_name}.pkl"),
                )

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
