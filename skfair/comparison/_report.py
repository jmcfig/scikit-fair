"""ComparisonReport class for fairness method comparison visualizations."""

from ._utils import validate_results_df, detect_metrics, classify_metric
from ._plots import (
    _plot_performance_bars,
    _plot_fairness_averaged,
    _plot_fairness_detailed,
    _plot_tradeoff_scatter,
    _summary_tables,
    _plot_ranking_heatmap,
)


class ComparisonReport:
    """Visualization report for fairness method comparison results.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns ``dataset``, ``method``, ``classifier``,
        plus one column per metric (e.g. ``accuracy``, ``spd``).
        Optional ``{metric}_std`` columns are preserved but not plotted.
    metrics : list of str, optional
        Explicit list of metric column names to use. When *None*,
        metrics are auto-detected from the DataFrame columns.

    Examples
    --------
    >>> report = ComparisonReport(results_df)
    >>> report.plot_performance()
    >>> report.plot_fairness_averaged(metric="spd")
    >>> tables = report.summary_tables()
    """

    def __init__(self, results_df, metrics=None, datasets=None, methods=None, classifiers=None):
        validate_results_df(results_df)
        self.df = results_df.copy()
        if datasets is not None:
            self.df = self.df[self.df["dataset"].isin(datasets)]
        if methods is not None:
            self.df = self.df[self.df["method"].isin(methods)]
        if classifiers is not None:
            self.df = self.df[self.df["classifier"].isin(classifiers)]
        self.datasets = sorted(self.df["dataset"].unique().tolist())
        self.methods = sorted(self.df["method"].unique().tolist())
        self.classifiers = sorted(self.df["classifier"].unique().tolist())
        self.metrics = metrics if metrics is not None else detect_metrics(self.df)
        self.performance_metrics = [m for m in self.metrics if classify_metric(m) == "performance"]
        self.fairness_metrics = [m for m in self.metrics if classify_metric(m) == "fairness"]

    def _resolve_datasets(self, datasets):
        return datasets if datasets is not None else self.datasets

    def _resolve_methods(self, methods):
        return methods if methods is not None else self.methods

    def _resolve_classifiers(self, classifiers):
        return classifiers if classifiers is not None else self.classifiers

    def _filter_df(self, datasets, methods, classifiers):
        """Return self.df filtered by datasets, methods, and classifiers."""
        df = self.df
        df = df[df["dataset"].isin(datasets)]
        df = df[df["method"].isin(methods)]
        df = df[df["classifier"].isin(classifiers)]
        return df

    def plot_performance(self, metrics=None, datasets=None, methods=None, classifiers=None, **kw):
        """Grouped bar charts of performance metrics.

        Returns (fig, axes).
        """
        metrics = metrics or self.performance_metrics
        datasets = self._resolve_datasets(datasets)
        methods = self._resolve_methods(methods)
        classifiers = self._resolve_classifiers(classifiers)
        df = self._filter_df(datasets, methods, classifiers)
        return _plot_performance_bars(df, metrics, datasets, **kw)

    def plot_fairness_averaged(self, metric="spd", datasets=None, methods=None, classifiers=None, **kw):
        """Bars averaged over classifiers for a single fairness metric.

        Returns (fig, axes).
        """
        datasets = self._resolve_datasets(datasets)
        methods = self._resolve_methods(methods)
        classifiers = self._resolve_classifiers(classifiers)
        df = self._filter_df(datasets, methods, classifiers)
        return _plot_fairness_averaged(df, metric, datasets, **kw)

    def plot_fairness_detailed(self, metric="spd", datasets=None, methods=None, classifiers=None, **kw):
        """Grouped bars per classifier for a single fairness metric.

        Returns (fig, axes).
        """
        datasets = self._resolve_datasets(datasets)
        methods = self._resolve_methods(methods)
        classifiers = self._resolve_classifiers(classifiers)
        df = self._filter_df(datasets, methods, classifiers)
        return _plot_fairness_detailed(df, metric, datasets, **kw)

    def plot_tradeoff(self, fairness_metric="spd", performance_metric="accuracy",
                      datasets=None, methods=None, classifiers=None, **kw):
        """Scatter plot: |fairness| vs performance.

        Returns (fig, axes).
        """
        datasets = self._resolve_datasets(datasets)
        methods = self._resolve_methods(methods)
        classifiers = self._resolve_classifiers(classifiers)
        df = self._filter_df(datasets, methods, classifiers)
        return _plot_tradeoff_scatter(df, fairness_metric, performance_metric,
                                      datasets, **kw)

    def plot_ranking(self, metrics=None, datasets=None, higher_is_better=None,
                     classifier=None, methods=None, classifiers=None, **kw):
        """Heatmap of method rankings per dataset.

        Parameters
        ----------
        classifier : None, "average", "best", or a classifier name.
            How to aggregate across classifiers before ranking.
        methods : list of str, optional
            Filter to these methods only.
        classifiers : list of str, optional
            Filter to these classifiers only.

        Returns (fig, axes).
        """
        metrics = metrics or self.metrics
        datasets = self._resolve_datasets(datasets)
        methods = self._resolve_methods(methods)
        classifiers = self._resolve_classifiers(classifiers)
        df = self._filter_df(datasets, methods, classifiers)
        return _plot_ranking_heatmap(df, metrics, datasets, higher_is_better,
                                     classifier=classifier, **kw)

    def summary_tables(self, metrics=None, datasets=None, classifier=None,
                       methods=None, classifiers=None):
        """Pivot tables of metric values per method.

        Parameters
        ----------
        classifier : None, "average", "best", or a classifier name.
            How to aggregate across classifiers.
        methods : list of str, optional
            Filter to these methods only.
        classifiers : list of str, optional
            Filter to these classifiers only.

        Returns dict[str, DataFrame].
        """
        metrics = metrics or self.metrics
        datasets = self._resolve_datasets(datasets)
        methods = self._resolve_methods(methods)
        classifiers = self._resolve_classifiers(classifiers)
        df = self._filter_df(datasets, methods, classifiers)
        return _summary_tables(df, metrics, datasets, classifier=classifier)

    def plot_all(self, datasets=None, fairness_metric="spd", methods=None, classifiers=None):
        """Run all 5 plot methods and return list of (fig, axes)."""
        datasets = self._resolve_datasets(datasets)
        fm = fairness_metric if fairness_metric in self.fairness_metrics else self.fairness_metrics[0]
        results = []
        results.append(self.plot_performance(datasets=datasets, methods=methods, classifiers=classifiers))
        if self.fairness_metrics:
            results.append(self.plot_fairness_averaged(
                metric=fm, datasets=datasets, methods=methods, classifiers=classifiers))
            results.append(self.plot_fairness_detailed(
                metric=fm, datasets=datasets, methods=methods, classifiers=classifiers))
        if self.fairness_metrics and self.performance_metrics:
            results.append(self.plot_tradeoff(
                fairness_metric=fm,
                performance_metric=self.performance_metrics[0],
                datasets=datasets, methods=methods, classifiers=classifiers))
        results.append(self.plot_ranking(datasets=datasets, methods=methods, classifiers=classifiers))
        return results
