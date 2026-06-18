"""Tests for skfair.experimentation.Experiment."""

import os

import numpy as np
import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from skfair.experimentation import Experiment
from skfair.experimentation._registry import (
    DEFAULT_METRICS,
    METHOD_REGISTRY,
)


def _make_toy_dataset(n=120, random_state=42):
    """Return (X, y) where X is a DataFrame with a 'group' sensitive attr."""
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame({
        "feat1": rng.randn(n),
        "feat2": rng.randn(n),
        "group": rng.choice([0, 1], size=n),
    })
    y = rng.choice([0, 1], size=n)
    return X, y


# ------------------------------------------------------------------ #
# Construction & validation
# ------------------------------------------------------------------ #

class TestConstruction:
    def test_default_construction(self):
        exp = Experiment()
        assert exp.dataset_names == ["adult"]
        assert exp.methods == list(METHOD_REGISTRY.keys())
        assert "LogReg" in exp.classifiers
        assert exp.metrics == list(DEFAULT_METRICS)
        assert exp.n_splits == 5
        assert exp.random_state == 42

    def test_custom_construction(self):
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline"],
            classifiers={"DT": DecisionTreeClassifier()},
            metrics=["accuracy"],
            n_splits=3,
            random_state=0,
        )
        assert exp.dataset_names == ["ricci"]
        assert exp.methods == ["Baseline"]
        assert "DT" in exp.classifiers
        assert exp.metrics == ["accuracy"]
        assert exp.n_splits == 3

    def test_unknown_dataset_warns(self):
        with pytest.warns(UserWarning, match="Unknown dataset"):
            exp = Experiment(datasets=["nonexistent"])
        assert exp.dataset_names == ["adult"]

    def test_unknown_method_warns(self):
        with pytest.warns(UserWarning, match="Unknown method"):
            exp = Experiment(methods=["NonExistent"])
        assert exp.methods == ["FairSmote"]

    def test_unknown_metric_warns(self):
        with pytest.warns(UserWarning, match="Unknown metric"):
            exp = Experiment(metrics=["nonexistent"])
        assert exp.metrics == []

    def test_classifier_dict_instances(self):
        clf = LogisticRegression()
        exp = Experiment(classifiers={"LR": clf})
        assert "LR" in exp.classifiers

    def test_classifier_list_of_dicts(self):
        spec = [{"path": "sklearn.linear_model.LogisticRegression", "name": "LR"}]
        exp = Experiment(classifiers=spec)
        assert "LR" in exp.classifiers

    def test_classifier_invalid_type(self):
        with pytest.raises(TypeError):
            Experiment(classifiers=42)

    def test_classifier_bad_import_warns(self):
        with pytest.warns(UserWarning, match="Cannot import"):
            exp = Experiment(classifiers=[{"path": "nonexistent.module.Cls"}])
        assert exp.classifiers == {}

    def test_from_config_classmethod(self):
        yml = """
datasets:
  - name: ricci
methods:
  - Baseline
cv:
  n_splits: 2
  random_state: 0
"""
        exp = Experiment.from_config(yml)
        assert exp.dataset_names == ["ricci"]
        assert exp.methods == ["Baseline"]
        assert exp.n_splits == 2

    def test_from_config_file(self, tmp_path):
        yml = """
datasets:
  - name: ricci
"""
        f = tmp_path / "config.yaml"
        f.write_text(yml)
        exp = Experiment.from_config(str(f))
        assert exp.dataset_names == ["ricci"]


# ------------------------------------------------------------------ #
# Pre-run errors
# ------------------------------------------------------------------ #

class TestPreRunErrors:
    def test_save_before_run(self):
        exp = Experiment()
        with pytest.raises(RuntimeError, match="No results"):
            exp.save("out.csv")

    def test_to_report_before_run(self):
        exp = Experiment()
        with pytest.raises(RuntimeError, match="No results"):
            exp.to_report()

    def test_get_auditor_before_run(self):
        exp = Experiment(audit_fairness=True)
        with pytest.raises(RuntimeError, match="Call .run()"):
            exp.get_fairness_auditor("x", "y", "z")

    def test_get_auditor_without_flag(self):
        exp = Experiment(audit_fairness=False)
        exp.results_ = pd.DataFrame()  # pretend run happened
        with pytest.raises(RuntimeError, match="not enabled"):
            exp.get_fairness_auditor("x", "y", "z")


# ------------------------------------------------------------------ #
# Integration tests (ricci — small & fast)
# ------------------------------------------------------------------ #

class TestIntegration:
    """Integration tests using the ricci dataset (118 samples)."""

    def test_run_baseline_only(self):
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline"],
            metrics=["accuracy"],
            n_splits=2,
        )
        df = exp.run(verbose=False)
        assert len(df) == 1
        assert "accuracy" in df.columns
        assert "accuracy_std" not in df.columns
        assert df.iloc[0]["method"] == "Baseline"

    def test_run_multiple_methods(self):
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline", "FairSmote", "Massaging"],
            metrics=["accuracy"],
            n_splits=2,
        )
        df = exp.run(verbose=False)
        assert len(df) == 3

    def test_run_multiple_classifiers(self):
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline"],
            classifiers={
                "LR": LogisticRegression(solver="liblinear", max_iter=1000),
                "DT": DecisionTreeClassifier(random_state=42),
            },
            metrics=["accuracy"],
            n_splits=2,
        )
        df = exp.run(verbose=False)
        assert len(df) == 2

    def test_run_single_split(self):
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline"],
            metrics=["accuracy"],
            n_splits=1,
        )
        df = exp.run(verbose=False)
        assert "accuracy_std" not in df.columns

    def test_run_with_audit_bias(self):
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline"],
            metrics=["accuracy"],
            n_splits=2,
            audit_bias=True,
        )
        exp.run(verbose=False)
        assert "ricci" in exp.bias_reports_

    def test_run_with_audit_fairness(self):
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline"],
            metrics=["accuracy"],
            n_splits=2,
            audit_fairness=True,
        )
        exp.run(verbose=False)
        auditor = exp.get_fairness_auditor("Ricci", "Baseline", "LogReg")
        assert auditor is not None

    def test_get_auditor_bad_key(self):
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline"],
            metrics=["accuracy"],
            n_splits=2,
            audit_fairness=True,
        )
        exp.run(verbose=False)
        with pytest.raises(KeyError):
            exp.get_fairness_auditor("Ricci", "NonExistent", "LogReg")

    def test_save_csv(self, tmp_path):
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline"],
            metrics=["accuracy"],
            n_splits=2,
        )
        exp.run(verbose=False)
        base = tmp_path / "out"
        exp.save(str(base), results_csv=True)
        csv_path = tmp_path / "out.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 1

    def test_save_pickle(self, tmp_path):
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline"],
            metrics=["accuracy"],
            n_splits=2,
        )
        exp.run(verbose=False)
        base = tmp_path / "out"
        exp.save(str(base), object_pkl=True)
        pkl_path = tmp_path / "out.pkl"
        assert pkl_path.exists()
        loaded = Experiment.load(str(pkl_path))
        assert len(loaded.results_) == 1

    def test_auto_save(self, tmp_path):
        base = tmp_path / "auto"
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline"],
            metrics=["accuracy"],
            n_splits=2,
            save_results_csv=True,
            save_object_pkl=True,
            save_path=str(base),
        )
        exp.run(verbose=False)
        assert (tmp_path / "auto.csv").exists()
        assert (tmp_path / "auto.pkl").exists()

    def test_save_models_all(self, tmp_path):
        base = tmp_path / "exp"
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline", "FairSmote"],
            metrics=["accuracy"],
            n_splits=2,
            save_models={"models": "all", "full_data_retrain": True},
            save_path=str(base),
        )
        exp.run(verbose=False)
        models_dir = tmp_path / "exp_models"
        assert models_dir.exists()
        pkl_files = list(models_dir.glob("*.pkl"))
        assert len(pkl_files) == 2
        assert len(exp.models_) == 2
        # Check that models are fitted pipelines
        for pipe in exp.models_.values():
            assert hasattr(pipe, "predict")

    def test_save_models_specific(self, tmp_path):
        base = tmp_path / "exp"
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline", "FairSmote"],
            metrics=["accuracy"],
            n_splits=2,
            save_models={
                "models": [{"method": "Baseline", "classifier": "LogReg"}],
                "full_data_retrain": True,
            },
            save_path=str(base),
        )
        exp.run(verbose=False)
        assert len(exp.models_) == 1
        key = list(exp.models_.keys())[0]
        assert key[1] == "Baseline"

    def test_save_models_last_fold(self, tmp_path):
        base = tmp_path / "exp"
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline"],
            metrics=["accuracy"],
            n_splits=2,
            save_models={"models": "all", "full_data_retrain": False},
            save_path=str(base),
        )
        exp.run(verbose=False)
        assert len(exp.models_) == 1
        for pipe in exp.models_.values():
            assert hasattr(pipe, "predict")

    def test_to_report(self):
        exp = Experiment(
            datasets=["ricci"],
            methods=["Baseline"],
            metrics=["accuracy"],
            n_splits=2,
        )
        exp.run(verbose=False)
        report = exp.to_report()
        from skfair.comparison import ComparisonReport
        assert isinstance(report, ComparisonReport)

    def test_from_example_config(self):
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "examples", "benchmark_config.yaml"
        )
        if not os.path.isfile(yaml_path):
            pytest.skip("example config not found")
        exp = Experiment.from_config(yaml_path)
        assert exp.dataset_names == ["ricci", "german"]
        assert len(exp.methods) == 12
        assert len(exp.classifiers) == 2
        assert exp.audit_bias is True
        assert exp.audit_fairness is True


# ------------------------------------------------------------------ #
# Custom dataset dicts
# ------------------------------------------------------------------ #

class TestCustomDatasets:
    """Tests for user-provided dataset dicts."""

    def test_custom_dataset_dict(self):
        X, y = _make_toy_dataset()
        exp = Experiment(
            datasets=[
                {"name": "toy", "data": (X, y), "sens_attr": "group"},
            ],
            methods=["Baseline"],
            metrics=["accuracy"],
            n_splits=2,
        )
        df = exp.run(verbose=False)
        assert len(df) == 1
        assert df.iloc[0]["dataset"] == "toy"
        assert "accuracy" in df.columns

    def test_mixed_registry_and_custom(self):
        X, y = _make_toy_dataset()
        exp = Experiment(
            datasets=[
                "ricci",
                {"name": "toy", "data": (X, y), "sens_attr": "group"},
            ],
            methods=["Baseline"],
            metrics=["accuracy"],
            n_splits=2,
        )
        df = exp.run(verbose=False)
        assert len(df) == 2
        assert set(df["dataset"]) == {"Ricci", "toy"}

    def test_custom_dataset_missing_keys(self):
        with pytest.raises(ValueError, match="missing required keys"):
            Experiment(datasets=[{"name": "bad"}])

    def test_custom_dataset_missing_name(self):
        X, y = _make_toy_dataset()
        with pytest.raises(ValueError, match="missing required keys"):
            Experiment(datasets=[{"data": (X, y), "sens_attr": "group"}])

    def test_custom_dataset_invalid_type(self):
        with pytest.raises(TypeError, match="str or dict"):
            Experiment(datasets=[42])
