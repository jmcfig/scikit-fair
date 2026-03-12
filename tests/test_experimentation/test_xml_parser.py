"""Tests for skfair.experimentation._xml_parser."""

import os
import xml.etree.ElementTree as ET

import pytest

from skfair.experimentation._xml_parser import _auto_cast, parse_experiment_xml


# ------------------------------------------------------------------ #
# _auto_cast
# ------------------------------------------------------------------ #

class TestAutoCast:
    @pytest.mark.parametrize("val,expected", [
        ("true", True), ("True", True), ("TRUE", True),
        ("false", False), ("False", False), ("FALSE", False),
    ])
    def test_auto_cast_true_false(self, val, expected):
        assert _auto_cast(val) is expected

    def test_auto_cast_int(self):
        assert _auto_cast("42") == 42
        assert isinstance(_auto_cast("42"), int)

    def test_auto_cast_float(self):
        assert _auto_cast("3.14") == pytest.approx(3.14)
        assert isinstance(_auto_cast("3.14"), float)

    def test_auto_cast_string(self):
        assert _auto_cast("liblinear") == "liblinear"
        assert isinstance(_auto_cast("liblinear"), str)


# ------------------------------------------------------------------ #
# parse_experiment_xml
# ------------------------------------------------------------------ #

class TestParseExperimentXml:
    def test_parse_minimal_xml(self):
        cfg = parse_experiment_xml("<experiment/>")
        assert set(cfg.keys()) == {"datasets", "methods", "classifiers", "cv", "audit", "metrics"}
        assert cfg["datasets"] == []
        assert cfg["methods"] == []
        assert cfg["classifiers"] == []
        assert cfg["cv"] == {}
        assert cfg["audit"] == {}
        assert cfg["metrics"] == []

    def test_parse_full_xml(self):
        xml = """
        <experiment>
          <datasets><dataset name="ricci"/></datasets>
          <methods><method name="Baseline"/></methods>
          <classifiers>
            <classifier path="sklearn.linear_model.LogisticRegression" solver="liblinear"/>
          </classifiers>
          <cv n_splits="5" random_state="42"/>
          <audit bias="true" fairness="false"/>
          <metrics><metric name="accuracy"/></metrics>
        </experiment>
        """
        cfg = parse_experiment_xml(xml)
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
        xml = '<experiment><datasets><dataset name="adult" sens_attr="race"/></datasets></experiment>'
        cfg = parse_experiment_xml(xml)
        assert cfg["datasets"][0]["sens_attr"] == "race"

    def test_parse_classifiers(self):
        xml = """
        <experiment><classifiers>
          <classifier path="sklearn.svm.SVC" name="MySVC" probability="true"/>
        </classifiers></experiment>
        """
        cfg = parse_experiment_xml(xml)
        clf = cfg["classifiers"][0]
        assert clf["path"] == "sklearn.svm.SVC"
        assert clf["name"] == "MySVC"
        assert clf["probability"] is True

    def test_parse_cv_attributes(self):
        xml = '<experiment><cv n_splits="10" random_state="0"/></experiment>'
        cfg = parse_experiment_xml(xml)
        assert cfg["cv"]["n_splits"] == 10
        assert isinstance(cfg["cv"]["n_splits"], int)
        assert cfg["cv"]["random_state"] == 0
        assert isinstance(cfg["cv"]["random_state"], int)

    def test_parse_audit_booleans(self):
        xml = '<experiment><audit bias="true" fairness="false"/></experiment>'
        cfg = parse_experiment_xml(xml)
        assert cfg["audit"]["bias"] is True
        assert cfg["audit"]["fairness"] is False

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_experiment_xml("/nonexistent/path/config.xml")

    def test_malformed_xml(self):
        with pytest.raises(ET.ParseError):
            parse_experiment_xml("<experiment><unclosed>")

    def test_parse_from_file(self, tmp_path):
        xml_content = '<experiment><datasets><dataset name="ricci"/></datasets></experiment>'
        f = tmp_path / "config.xml"
        f.write_text(xml_content)
        cfg = parse_experiment_xml(str(f))
        assert cfg["datasets"][0]["name"] == "ricci"

    def test_example_config_xml(self):
        xml_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "examples", "experiment_config.xml"
        )
        if not os.path.isfile(xml_path):
            pytest.skip("example config not found")
        cfg = parse_experiment_xml(xml_path)
        assert len(cfg["datasets"]) == 1
        assert cfg["datasets"][0]["name"] == "ricci"
        assert len(cfg["methods"]) == 12
        assert len(cfg["classifiers"]) == 2
