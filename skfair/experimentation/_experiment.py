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
from ._xml_parser import parse_experiment_xml
from ._runner import build_pipeline, run_cv


class Experiment:
    """Run a fairness-method comparison experiment.

    Parameters
    ----------
    datasets : list of str, optional
        Dataset names (keys of ``DATASET_REGISTRY``).  Default: ``["adult"]``.
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
    audit_bias : bool
        If *True*, create a ``BiasAuditor`` per dataset after loading.
    audit_fairness : bool
        If *True*, store out-of-fold predictions so that
        :meth:`audit_fairness` can build a ``FairnessAuditor`` later.
    xml : str, optional
        Path to (or raw string of) an XML configuration.  When provided,
        all other arguments are **ignored** and the config is read from XML.
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
        audit_bias=False,
        audit_fairness=False,
        save_results=False,
        save_object=False,
        save_path="experiment",
        xml=None,
    ):
        if xml is not None:
            self._init_from_xml(xml)
            return

        # -- datasets --
        self.datasets = self._validate_datasets(datasets or ["adult"])

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
        self.audit_bias = audit_bias
        self.audit_fairness = audit_fairness
        self.save_results = save_results
        self.save_object = save_object
        self.save_path = save_path

        # Populated by run()
        self.results_ = None
        self.bias_reports_ = {}
        self._predictions = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_xml(cls, path):
        """Create an ``Experiment`` from an XML configuration file/string."""
        return cls(xml=path)

    def _init_from_xml(self, source):
        """Parse XML config and set attributes."""
        cfg = parse_experiment_xml(source)

        # datasets
        ds_names = [d["name"] for d in cfg["datasets"]] if cfg["datasets"] else ["adult"]
        self.datasets = self._validate_datasets(ds_names)
        # build dataset_config from XML attributes
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

        # audit
        self.audit_bias = cfg["audit"].get("bias", False)
        self.audit_fairness = cfg["audit"].get("fairness", False)

        # save (defaults off from XML; user can override after from_xml())
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
    def _validate_datasets(names):
        validated = []
        for name in names:
            key = name.lower()
            if key not in DATASET_REGISTRY:
                warnings.warn(
                    f"Unknown dataset '{name}', falling back to 'adult'.",
                    stacklevel=2,
                )
                key = "adult"
            validated.append(key)
        return validated

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
        """Accept a dict of instances or a list of XML-style dicts/strings."""
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
            One row per (dataset, method, classifier) with
            ``{metric}_mean`` / ``{metric}_std`` columns.
        """
        # Resolve metric callables
        metric_fns = {}
        metric_types = {}
        for name in self.metrics:
            info = METRIC_REGISTRY[name]
            metric_fns[name] = _import_object(info["path"])
            metric_types[name] = info["type"]

        rows = []

        for ds_key in self.datasets:
            ds_info = {**DATASET_REGISTRY[ds_key]}
            # Apply per-dataset overrides
            ds_info.update(self.dataset_config.get(ds_key, {}))

            loader = _import_object(ds_info["loader"])
            X, y = loader()
            sens_attr = ds_info["sens_attr"]
            priv_group = ds_info.get("priv_group", 1)

            # Friendly display name
            ds_display = ds_key.replace("_", " ").title()

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
                            acc = cv_result.get("accuracy_mean", float("nan"))
                            spd = cv_result.get("spd_mean", float("nan"))
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
                            row[f"{m}_mean"] = float("nan")
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
