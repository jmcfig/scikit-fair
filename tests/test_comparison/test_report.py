"""Tests for ComparisonReport."""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from skfair.comparison import ComparisonReport


@pytest.fixture
def results_df():
    """Synthetic results: 2 datasets x 3 methods x 2 classifiers x 2 metrics."""
    rows = []
    rng = np.random.RandomState(0)
    for ds in ["adult", "compas"]:
        for method in ["Massaging", "FairSmote", "Reweighing"]:
            for clf in ["ClfA", "ClfB"]:
                rows.append({
                    "dataset": ds,
                    "method": method,
                    "classifier": clf,
                    "accuracy": rng.uniform(0.7, 0.95),
                    "spd": rng.uniform(-0.15, 0.15),
                })
    return pd.DataFrame(rows)


# ---- Construction ----

class TestInit:
    def test_init_basic(self, results_df):
        report = ComparisonReport(results_df)
        assert set(report.datasets) == {"adult", "compas"}
        assert set(report.methods) == {"Massaging", "FairSmote", "Reweighing"}
        assert set(report.classifiers) == {"ClfA", "ClfB"}
        assert "accuracy" in report.metrics
        assert "spd" in report.metrics
        assert report.performance_metrics == ["accuracy"]
        assert report.fairness_metrics == ["spd"]

    def test_init_filter_datasets(self, results_df):
        report = ComparisonReport(results_df, datasets=["adult"])
        assert report.datasets == ["adult"]
        assert len(report.df) == 6  # 3 methods x 2 classifiers

    def test_init_filter_methods(self, results_df):
        report = ComparisonReport(results_df, methods=["Massaging", "FairSmote"])
        assert set(report.methods) == {"Massaging", "FairSmote"}
        assert len(report.df) == 8  # 2 datasets x 2 methods x 2 classifiers

    def test_init_filter_classifiers(self, results_df):
        report = ComparisonReport(results_df, classifiers=["ClfA"])
        assert report.classifiers == ["ClfA"]
        assert len(report.df) == 6  # 2 datasets x 3 methods x 1 classifier


# ---- Summary tables ----

class TestSummaryTables:
    def test_summary_tables_default(self, results_df):
        report = ComparisonReport(results_df)
        tables = report.summary_tables()
        assert isinstance(tables, dict)
        assert set(tables.keys()) == {"adult", "compas"}
        for ds, tbl in tables.items():
            assert isinstance(tbl, pd.DataFrame)
            assert set(tbl.index) == {"Massaging", "FairSmote", "Reweighing"}

    def test_summary_tables_classifier_best(self, results_df):
        report = ComparisonReport(results_df)
        tables = report.summary_tables(classifier="best")
        assert "adult" in tables
        assert tables["adult"].shape[0] == 3

    def test_summary_tables_classifier_name(self, results_df):
        report = ComparisonReport(results_df)
        tables = report.summary_tables(classifier="ClfA")
        assert "adult" in tables
        assert tables["adult"].shape[0] == 3


# ---- Plot tests ----

class TestPlots:
    def test_plot_metric_bar(self, results_df):
        report = ComparisonReport(results_df)
        fig, axes = report.plot_metric_bar(metric="accuracy")
        assert fig is not None
        assert axes is not None
        plt.close("all")

    def test_plot_metric_bar_with_filters(self, results_df):
        report = ComparisonReport(results_df)
        fig, axes = report.plot_metric_bar(
            metric="accuracy",
            datasets=["adult"],
            methods=["Massaging"],
            classifiers=["ClfA"],
        )
        assert fig is not None
        plt.close("all")

    def test_plot_tradeoff(self, results_df):
        report = ComparisonReport(results_df)
        fig, axes = report.plot_tradeoff(
            fairness_metric="spd", performance_metric="accuracy"
        )
        assert fig is not None
        plt.close("all")

    def test_plot_ranking(self, results_df):
        report = ComparisonReport(results_df)
        fig, axes = report.plot_ranking()
        assert fig is not None
        plt.close("all")

    def test_plot_ranking_classifier_modes(self, results_df):
        report = ComparisonReport(results_df)
        for mode in ["average", "best", "ClfA"]:
            fig, axes = report.plot_ranking(classifier=mode)
            assert fig is not None
            plt.close("all")

    def test_plot_all(self, results_df):
        report = ComparisonReport(results_df)
        result = report.plot_all()
        # accuracy bar + spd bar + tradeoff + ranking = 4
        assert isinstance(result, list)
        assert len(result) == 4
        for fig, axes in result:
            assert fig is not None
        plt.close("all")


# ---- HTML report ----

class TestToHtml:
    def test_to_html(self, results_df):
        report = ComparisonReport(results_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            report.to_html(path)
            assert os.path.exists(path)
            content = open(path).read()
            assert len(content) > 100
            assert "<html" in content.lower()
            assert "adult" in content
        plt.close("all")

    def test_to_html_with_filters(self, results_df):
        report = ComparisonReport(results_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "filtered.html")
            report.to_html(
                path,
                datasets=["compas"],
                methods=["Massaging", "FairSmote"],
                classifiers=["ClfA"],
            )
            assert os.path.exists(path)
            content = open(path).read()
            assert len(content) > 100
        plt.close("all")
