"""Tests for skfair.experimentation._config_parser."""

import os

import pytest
import yaml

from skfair.experimentation._config_parser import parse_experiment_config


class TestParseExperimentConfig:
    def test_parse_minimal_yaml(self):
        cfg = parse_experiment_config("---\n")
        assert set(cfg.keys()) == {"datasets", "methods", "classifiers", "cv", "audit", "metrics"}
        assert cfg["datasets"] == []
        assert cfg["methods"] == []
        assert cfg["classifiers"] == []
        assert cfg["cv"] == {}
        assert cfg["audit"] == {}
        assert cfg["metrics"] == []

    def test_parse_full_yaml(self):
        yml = """
datasets:
  - name: ricci
methods:
  - name: Baseline
classifiers:
  - path: sklearn.linear_model.LogisticRegression
    solver: liblinear
cv:
  n_splits: 5
  random_state: 42
audit:
  bias: true
  fairness: false
metrics:
  - name: accuracy
"""
        cfg = parse_experiment_config(yml)
        assert len(cfg["datasets"]) == 1
        assert cfg["datasets"][0]["name"] == "ricci"
        assert len(cfg["methods"]) == 1
        assert len(cfg["classifiers"]) == 1
        assert cfg["cv"]["n_splits"] == 5
        assert cfg["cv"]["random_state"] == 42
        assert cfg["audit"]["bias"] is True
        assert cfg["audit"]["fairness"] is False
        assert len(cfg["metrics"]) == 1

    def test_parse_datasets_with_overrides(self):
        yml = """
datasets:
  - name: adult
    sens_attr: race
"""
        cfg = parse_experiment_config(yml)
        assert cfg["datasets"][0]["sens_attr"] == "race"

    def test_parse_classifiers(self):
        yml = """
classifiers:
  - path: sklearn.svm.SVC
    name: MySVC
    probability: true
"""
        cfg = parse_experiment_config(yml)
        clf = cfg["classifiers"][0]
        assert clf["path"] == "sklearn.svm.SVC"
        assert clf["name"] == "MySVC"
        assert clf["probability"] is True

    def test_parse_cv_attributes(self):
        yml = """
cv:
  n_splits: 10
  random_state: 0
"""
        cfg = parse_experiment_config(yml)
        assert cfg["cv"]["n_splits"] == 10
        assert isinstance(cfg["cv"]["n_splits"], int)
        assert cfg["cv"]["random_state"] == 0
        assert isinstance(cfg["cv"]["random_state"], int)

    def test_parse_audit_booleans(self):
        yml = """
audit:
  bias: true
  fairness: false
"""
        cfg = parse_experiment_config(yml)
        assert cfg["audit"]["bias"] is True
        assert cfg["audit"]["fairness"] is False

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_experiment_config("/nonexistent/path/config.yaml")

    def test_malformed_yaml(self):
        with pytest.raises(yaml.YAMLError):
            parse_experiment_config(":\n  - :\n    - : [\n")

    def test_parse_from_file(self, tmp_path):
        yml_content = """
datasets:
  - name: ricci
"""
        f = tmp_path / "config.yaml"
        f.write_text(yml_content)
        cfg = parse_experiment_config(str(f))
        assert cfg["datasets"][0]["name"] == "ricci"

    def test_example_config(self):
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "examples", "benchmark_config.yaml"
        )
        if not os.path.isfile(yaml_path):
            pytest.skip("example config not found")
        cfg = parse_experiment_config(yaml_path)
        assert len(cfg["datasets"]) == 2
        assert cfg["datasets"][0]["name"] == "ricci"
        assert len(cfg["methods"]) == 12
        assert len(cfg["classifiers"]) == 2

    def test_shorthand_methods(self):
        yml = """
methods:
  - Baseline
  - FairSmote
"""
        cfg = parse_experiment_config(yml)
        assert cfg["methods"][0] == {"name": "Baseline"}
        assert cfg["methods"][1] == {"name": "FairSmote"}

    def test_shorthand_datasets(self):
        yml = """
datasets:
  - ricci
  - german
"""
        cfg = parse_experiment_config(yml)
        assert cfg["datasets"][0] == {"name": "ricci"}
        assert cfg["datasets"][1] == {"name": "german"}
