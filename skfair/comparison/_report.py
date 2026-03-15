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

    def to_html(self, path, datasets=None, methods=None, classifiers=None,
                metrics=None, fairness_metric="spd", performance_metric="accuracy",
                classifier=None):
        """Export an HTML report with embedded matplotlib charts.

        Parameters
        ----------
        path : str
            Output file path (e.g. ``"report.html"``).
        datasets, methods, classifiers : list of str, optional
            Filters.
        metrics : list of str, optional
            Metrics to include. Defaults to all detected.
        fairness_metric : str
            Fairness metric for averaged/detailed/tradeoff charts.
        performance_metric : str
            Performance metric for tradeoff chart.
        classifier : None, "average", "best", or a classifier name
            How to aggregate for rankings/tables.
        """
        import io
        import base64
        import matplotlib.pyplot as plt
        from ._html_template import render_html_report

        datasets = self._resolve_datasets(datasets)
        methods = self._resolve_methods(methods)
        classifiers_list = self._resolve_classifiers(classifiers)
        metrics = metrics or self.metrics
        perf_metrics = [m for m in metrics if classify_metric(m) == "performance"]
        fair_metrics = [m for m in metrics if classify_metric(m) == "fairness"]

        pm = performance_metric if performance_metric in perf_metrics else (perf_metrics[0] if perf_metrics else None)

        def _fig_to_img(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            return f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%">'

        filtered_df = self._filter_df(datasets, methods, classifiers_list)

        # --- Performance charts: {metric: {dataset: img}} ---
        perf_charts = {}
        for m in perf_metrics:
            perf_charts[m] = {}
            for ds in datasets:
                ds_df = filtered_df[filtered_df["dataset"] == ds]
                fig, _ = _plot_performance_bars(ds_df, [m], [ds])
                perf_charts[m][ds] = _fig_to_img(fig)

        # --- Fairness charts: {metric: {dataset: img}} ---
        fair_charts = {}
        for m in fair_metrics:
            fair_charts[m] = {}
            for ds in datasets:
                ds_df = filtered_df[filtered_df["dataset"] == ds]
                fig, _ = _plot_fairness_detailed(ds_df, m, [ds])
                fair_charts[m][ds] = _fig_to_img(fig)

        # --- Ranking charts: {dataset: {agg: img}} ---
        rank_charts = {}
        for ds in datasets:
            agg_imgs = {}
            for agg_mode in ["average", "best"]:
                fig, _ = _plot_ranking_heatmap(filtered_df, metrics, [ds],
                                                classifier=agg_mode)
                label = agg_mode.capitalize()
                agg_imgs[label] = _fig_to_img(fig)
            for clf in classifiers_list:
                fig, _ = _plot_ranking_heatmap(filtered_df, metrics, [ds],
                                                classifier=clf)
                agg_imgs[clf] = _fig_to_img(fig)
            rank_charts[ds] = agg_imgs

        # --- Tradeoff charts: {fairness_metric: img} ---
        tradeoff_charts = {}
        if fair_metrics and pm:
            for fm in fair_metrics:
                fig, _ = self.plot_tradeoff(fairness_metric=fm, performance_metric=pm,
                                            datasets=datasets, methods=methods,
                                            classifiers=classifiers)
                tradeoff_charts[fm] = _fig_to_img(fig)

        # --- Tables: {dataset: {agg: df}} ---
        tables_avg = _summary_tables(filtered_df, metrics, datasets, classifier="average")
        tables_best = _summary_tables(filtered_df, metrics, datasets, classifier="best")
        tables = {}
        for ds in datasets:
            tables[ds] = {}
            if ds in tables_avg:
                tables[ds]["Average"] = tables_avg[ds]
            if ds in tables_best:
                tables[ds]["Best"] = tables_best[ds]
            for clf_name in classifiers_list:
                clf_tables = _summary_tables(filtered_df, metrics, datasets, classifier=clf_name)
                if ds in clf_tables:
                    tables[ds][clf_name] = clf_tables[ds]

        metadata = {
            "n_datasets": len(datasets),
            "n_methods": len(methods),
            "n_classifiers": len(classifiers_list),
            "n_metrics": len(metrics),
        }

        html = render_html_report(
            perf_charts=perf_charts,
            fair_charts=fair_charts,
            rank_charts=rank_charts,
            tradeoff_charts=tradeoff_charts,
            tables=tables,
            metadata=metadata,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

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
