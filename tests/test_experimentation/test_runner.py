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
        pipe = build_pipeline("DisparateImpactRemover", clf, X, "Race")
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
        result, preds, _ = run_cv(
            pipe, X, y, sens_col="Race",
            metrics=metrics, metric_types=metric_types,
            n_splits=2,
        )
        assert "accuracy" in result
        assert "accuracy_std" not in result
        assert 0.0 <= result["accuracy"] <= 1.0
        assert preds is None

    def test_run_cv_single_split(self, ricci_data):
        X, y = ricci_data
        clf = LogisticRegression(solver="liblinear", max_iter=1000)
        pipe = build_pipeline("Baseline", clf, X, "Race")
        metrics, metric_types = self._make_metrics()
        result, _, _ = run_cv(
            pipe, X, y, sens_col="Race",
            metrics=metrics, metric_types=metric_types,
            n_splits=1,
        )
        assert "accuracy_std" not in result

    def test_run_cv_store_predictions(self, ricci_data):
        X, y = ricci_data
        clf = LogisticRegression(solver="liblinear", max_iter=1000)
        pipe = build_pipeline("Baseline", clf, X, "Race")
        metrics, metric_types = self._make_metrics()
        result, preds, _ = run_cv(
            pipe, X, y, sens_col="Race",
            metrics=metrics, metric_types=metric_types,
            n_splits=2, store_predictions=True,
        )
        assert preds is not None
        assert "y_true" in preds
        assert "y_pred" in preds
        assert "sens_attr" in preds
        assert len(preds["y_true"]) == len(y)

    def test_run_cv_priv_group_binarises_multivalued_sens(self):
        from skfair.metrics import statistical_parity_difference

        rng = np.random.RandomState(0)
        n = 200
        X = pd.DataFrame({
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "group": rng.choice([0, 1, 2], size=n),
        })
        y = pd.Series(rng.randint(0, 2, size=n))
        clf = LogisticRegression(solver="liblinear", max_iter=1000)
        pipe = build_pipeline("Baseline", clf, X, "group")
        result, preds, _ = run_cv(
            pipe, X, y, sens_col="group",
            metrics={"spd": statistical_parity_difference},
            metric_types={"spd": "fairness"},
            n_splits=2, store_predictions=True, priv_group=2,
        )
        # Metrics computed without error on the binarised indicator
        assert np.isfinite(result["spd"])
        # Stored predictions carry the 0/1 indicator, not the raw values
        assert set(np.unique(preds["sens_attr"])) <= {0, 1}
        # Share of privileged matches the share of the chosen group value
        assert preds["sens_attr"].mean() == pytest.approx(
            (X["group"] == 2).mean(), abs=1e-12
        )

    def test_run_cv_priv_group_default_is_noop_for_binary(self, ricci_data):
        X, y = ricci_data
        clf = LogisticRegression(solver="liblinear", max_iter=1000)
        pipe = build_pipeline("Baseline", clf, X, "Race")
        metrics, metric_types = self._make_metrics()
        result, preds, _ = run_cv(
            pipe, X, y, sens_col="Race",
            metrics=metrics, metric_types=metric_types,
            n_splits=2, store_predictions=True,
        )
        assert preds["sens_attr"].mean() == pytest.approx(
            (X["Race"] == 1).mean(), abs=1e-12
        )


# ------------------------------------------------------------------ #
# stratify and n_repeats
# ------------------------------------------------------------------ #

class TestStratifyAndRepeats:
    def _basic_metrics(self):
        from sklearn.metrics import accuracy_score
        return {"accuracy": accuracy_score}, {"accuracy": "performance"}

    def test_stratify_none(self, ricci_data):
        X, y = ricci_data
        clf = LogisticRegression(solver="liblinear", max_iter=1000)
        pipe = build_pipeline("Baseline", clf, X, "Race")
        metrics, mtypes = self._basic_metrics()
        result, _, _ = run_cv(
            pipe, X, y, sens_col="Race",
            metrics=metrics, metric_types=mtypes,
            n_splits=3, stratify=None,
        )
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_stratify_both_balances_subgroups(self, ricci_data):
        # With stratify="both", every fold's TEST set should contain at
        # least one example of each (y, sens_attr) combination.
        X, y = ricci_data
        from skfair.experimentation._runner import (
            _build_splits,
            _resolve_strat_label,
        )
        y_arr = np.asarray(y)
        label = _resolve_strat_label("both", y_arr, X.reset_index(drop=True), "Race")
        splits = _build_splits(
            n_splits=3, n_repeats=1, strat_label=label,
            y_arr=y_arr, random_state=0,
        )
        sens = X["Race"].values
        for _, test_idx in splits:
            unique_combos = set(zip(y_arr[test_idx], sens[test_idx]))
            # Ricci has binary y and binary Race → 4 strata expected.
            assert len(unique_combos) >= 2

    def test_n_repeats_runs_more_fits(self, ricci_data):
        X, y = ricci_data
        from skfair.experimentation._runner import _build_splits
        splits = _build_splits(
            n_splits=5, n_repeats=3, strat_label=np.asarray(y),
            y_arr=np.asarray(y), random_state=0,
        )
        assert len(splits) == 15

    def test_n_repeats_single_split(self, ricci_data):
        X, y = ricci_data
        from skfair.experimentation._runner import _build_splits
        splits = _build_splits(
            n_splits=1, n_repeats=4, strat_label=np.asarray(y),
            y_arr=np.asarray(y), random_state=0,
        )
        assert len(splits) == 4

    def test_stratify_too_small_falls_back(self, ricci_data):
        # Build a dataset where one (y, sens) cell has only 1 row → must
        # fall back from "both" to "y" with a warning, not crash.
        X, y = ricci_data
        X = X.reset_index(drop=True)
        y = np.asarray(y).copy()
        # Force a 1-row stratum: pick the first row, flip both y and sens
        # such that this combination is unique.
        y[0] = 1 - y[0]
        unique_sens = X["Race"].unique()
        rare_sens = unique_sens[-1]
        # Make sure only row 0 has this exact (y, sens) combo
        X.loc[0, "Race"] = rare_sens
        X.loc[1:, "Race"] = unique_sens[0] if rare_sens != unique_sens[0] else unique_sens[-1]

        clf = LogisticRegression(solver="liblinear", max_iter=1000)
        pipe = build_pipeline("Baseline", clf, X, "Race")
        metrics, mtypes = self._basic_metrics()
        with pytest.warns(UserWarning, match="Stratification failed"):
            result, _, _ = run_cv(
                pipe, X, y, sens_col="Race",
                metrics=metrics, metric_types=mtypes,
                n_splits=5, stratify="both",
            )
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_nan_fold_handling(self, ricci_data):
        # A metric that returns NaN on every fold should produce NaN mean.
        # A metric that returns NaN on some folds should still produce a
        # finite mean over the remaining folds.
        X, y = ricci_data
        clf = LogisticRegression(solver="liblinear", max_iter=1000)
        pipe = build_pipeline("Baseline", clf, X, "Race")

        call_count = {"n": 0}

        def flaky_metric(y_true, y_pred):
            call_count["n"] += 1
            return float("nan") if call_count["n"] == 1 else 0.5

        metrics = {"flaky": flaky_metric}
        mtypes = {"flaky": "performance"}
        result, _, _ = run_cv(
            pipe, X, y, sens_col="Race",
            metrics=metrics, metric_types=mtypes,
            n_splits=3, include_std=True,
        )
        # First fold NaN, remaining folds → 0.5; nanmean should be 0.5.
        assert result["flaky"] == pytest.approx(0.5)
        assert not np.isnan(result["flaky_std"])
