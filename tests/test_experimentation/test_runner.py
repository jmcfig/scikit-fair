"""Tests for skfair.experimentation._runner."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from skfair.experimentation._runner import build_pipeline, run_cv
from skfair.experimentation._registry import METHOD_REGISTRY, _import_object
from skfair.datasets import load_ricci


@pytest.fixture(scope="module")
def ricci_data():
    """Load ricci once for the module."""
    X, y = load_ricci()
    return X, y


# ------------------------------------------------------------------ #
# build_pipeline
# ------------------------------------------------------------------ #

class TestBuildPipeline:
    def test_build_baseline_pipeline(self, ricci_data):
        X, _ = ricci_data
        clf = LogisticRegression(solver="liblinear")
        pipe = build_pipeline("Baseline", clf, X, "Race")
        assert len(pipe.steps) == 1
        assert pipe.steps[0][0] == "clf"

    def test_build_sampler_pipeline(self, ricci_data):
        X, _ = ricci_data
        clf = LogisticRegression(solver="liblinear")
        pipe = build_pipeline("FairSmote", clf, X, "Race")
        assert len(pipe.steps) == 2
        assert pipe.steps[0][0] == "method"
        assert pipe.steps[1][0] == "clf"

    def test_build_repair_pipeline(self, ricci_data):
        X, _ = ricci_data
        clf = LogisticRegression(solver="liblinear")
        pipe = build_pipeline("GeometricFairnessRepair", clf, X, "Race")
        assert len(pipe.steps) == 2
        method = pipe.steps[0][1]
        assert hasattr(method, "repair_columns")

    def test_build_meta_pipeline(self, ricci_data):
        X, _ = ricci_data
        clf = LogisticRegression(solver="liblinear")
        pipe = build_pipeline("ReweighingClassifier", clf, X, "Race")
        assert len(pipe.steps) == 1
        assert pipe.steps[0][0] == "clf"
        assert hasattr(pipe.steps[0][1], "estimator")

    def test_build_unknown_category(self, ricci_data):
        X, _ = ricci_data
        clf = LogisticRegression(solver="liblinear")
        # Temporarily inject a bad category into a method that has a real path
        original = METHOD_REGISTRY["FairSmote"]["category"]
        METHOD_REGISTRY["FairSmote"]["category"] = "unknown_cat"
        try:
            with pytest.raises(ValueError, match="Unknown category"):
                build_pipeline("FairSmote", clf, X, "Race")
        finally:
            METHOD_REGISTRY["FairSmote"]["category"] = original


# ------------------------------------------------------------------ #
# run_cv
# ------------------------------------------------------------------ #

class TestRunCv:
    def _make_metrics(self):
        from sklearn.metrics import accuracy_score
        return (
            {"accuracy": accuracy_score},
            {"accuracy": "performance"},
        )

    def test_run_cv_basic(self, ricci_data):
        X, y = ricci_data
        clf = LogisticRegression(solver="liblinear", max_iter=1000)
        pipe = build_pipeline("Baseline", clf, X, "Race")
        metrics, metric_types = self._make_metrics()
        result, preds = run_cv(
            pipe, X, y, sens_col="Race",
            metrics=metrics, metric_types=metric_types,
            n_splits=2,
        )
        assert "accuracy_mean" in result
        assert "accuracy_std" in result
        assert 0.0 <= result["accuracy_mean"] <= 1.0
        assert preds is None

    def test_run_cv_single_split(self, ricci_data):
        X, y = ricci_data
        clf = LogisticRegression(solver="liblinear", max_iter=1000)
        pipe = build_pipeline("Baseline", clf, X, "Race")
        metrics, metric_types = self._make_metrics()
        result, _ = run_cv(
            pipe, X, y, sens_col="Race",
            metrics=metrics, metric_types=metric_types,
            n_splits=1,
        )
        assert result["accuracy_std"] == 0.0

    def test_run_cv_store_predictions(self, ricci_data):
        X, y = ricci_data
        clf = LogisticRegression(solver="liblinear", max_iter=1000)
        pipe = build_pipeline("Baseline", clf, X, "Race")
        metrics, metric_types = self._make_metrics()
        result, preds = run_cv(
            pipe, X, y, sens_col="Race",
            metrics=metrics, metric_types=metric_types,
            n_splits=2, store_predictions=True,
        )
        assert preds is not None
        assert "y_true" in preds
        assert "y_pred" in preds
        assert "sens_attr" in preds
        assert len(preds["y_true"]) == len(y)
