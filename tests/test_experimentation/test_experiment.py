"""Tests for skfair.experimentation.Experiment."""

import os
import warnings

import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from skfair.experimentation import Experiment
from skfair.experimentation._registry import METHOD_REGISTRY, METRIC_REGISTRY


# ------------------------------------------------------------------ #
# Construction & validation
# ------------------------------------------------------------------ #

class TestConstruction:
    def test_default_construction(self):
        exp = Experiment()
        assert exp.datasets == ["adult"]
        assert exp.methods == list(METHOD_REGISTRY.keys())
        assert "LogReg" in exp.classifiers
        assert exp.metrics == list(METRIC_REGISTRY.keys())
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
        assert exp.datasets == ["ricci"]
        assert exp.methods == ["Baseline"]
        assert "DT" in exp.classifiers
        assert exp.metrics == ["accuracy"]
        assert exp.n_splits == 3

    def test_unknown_dataset_warns(self):
        with pytest.warns(UserWarning, match="Unknown dataset"):
            exp = Experiment(datasets=["nonexistent"])
        assert exp.datasets == ["adult"]

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

    def test_from_xml_classmethod(self):
        xml = """
        <experiment>
          <datasets><dataset name="ricci"/></datasets>
          <methods><method name="Baseline"/></methods>
          <cv n_splits="2" random_state="0"/>
        </experiment>
        """
        exp = Experiment.from_xml(xml)
        assert exp.datasets == ["ricci"]
        assert exp.methods == ["Baseline"]
        assert exp.n_splits == 2

    def test_from_xml_file(self, tmp_path):
        xml = '<experiment><datasets><dataset name="ricci"/></datasets></experiment>'
        f = tmp_path / "config.xml"
        f.write_text(xml)
        exp = Experiment.from_xml(str(f))
        assert exp.datasets == ["ricci"]


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
        assert "accuracy_mean" in df.columns
        assert "accuracy_std" in df.columns
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
        assert df.iloc[0]["accuracy_std"] == 0.0

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
        exp.save(str(base), results=True)
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
        exp.save(str(base), object=True)
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
            save_results=True,
            save_object=True,
            save_path=str(base),
        )
        exp.run(verbose=False)
        assert (tmp_path / "auto.csv").exists()
        assert (tmp_path / "auto.pkl").exists()

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

    def test_from_example_xml(self):
        xml_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "examples", "experiment_config.xml"
        )
        if not os.path.isfile(xml_path):
            pytest.skip("example config not found")
        exp = Experiment.from_xml(xml_path)
        assert exp.datasets == ["ricci"]
        assert len(exp.methods) == 12
        assert len(exp.classifiers) == 2
        assert exp.audit_bias is True
        assert exp.audit_fairness is True
