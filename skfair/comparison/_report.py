"""ComparisonReport class for fairness method comparison visualizations."""

from ._utils import validate_results_df, detect_metrics, classify_metric
from ._plots import (
    _plot_metric_bar,
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
    >>> report.plot_metric_bar(metric="accuracy")
    >>> report.plot_metric_bar(metric="spd")
    >>> tables = report.summary_tables()
    """

    def __init__(self, results_df, metrics=None, datasets=None, methods=None, classifiers=None):
        """
        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame with columns ``dataset``, ``method``, ``classifier``,
            plus one column per metric.
        metrics : list of str, optional
            Explicit list of metric column names to use. Auto-detected when
            *None*.
        datasets : list of str, optional
            Keep only these datasets. *None* keeps all.
        methods : list of str, optional
            Keep only these methods. *None* keeps all.
        classifiers : list of str, optional
            Keep only these classifiers. *None* keeps all.
        """
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

    def plot_metric_bar(self, metric=None, datasets=None, methods=None,
                        classifiers=None, reference_line="auto", show_std=False,
                        **kw):
        """Grouped bar chart for a single metric across datasets.

        Parameters
        ----------
        metric : str, optional
            Metric to plot. Defaults to first performance metric.
        datasets : list of str, optional
            Datasets to include. *None* uses all.
        methods : list of str, optional
            Methods to include. *None* uses all.
        classifiers : list of str, optional
            Classifiers to include. *None* uses all.
        reference_line : float, "auto", or None
            "auto" adds reference lines for fairness metrics.
        show_std : bool, default False
            Draw ``{metric}_std`` error bars (requires the std column).

        Returns
        -------
        tuple of (fig, axes)
        """
        metric = metric or (self.performance_metrics[0] if self.performance_metrics
                            else self.metrics[0])
        datasets = self._resolve_datasets(datasets)
        methods = self._resolve_methods(methods)
        classifiers = self._resolve_classifiers(classifiers)
        df = self._filter_df(datasets, methods, classifiers)
        return _plot_metric_bar(df, metric, datasets, reference_line=reference_line,
                                show_std=show_std, **kw)

    def plot_tradeoff(self, fairness_metric="spd", performance_metric="accuracy",
                      datasets=None, methods=None, classifiers=None, show_std=False,
                      **kw):
        """Scatter plot of fairness distance from ideal vs performance.

        The x-axis transforms the fairness metric so that **lower = fairer**
        regardless of metric family:

        - difference metrics (ideal = 0, e.g. ``spd``, ``eod``) plot as
          ``|metric|``;
        - ratio metrics (ideal = 1, e.g. ``disparate_impact``,
          ``equal_opportunity_ratio``) plot as ``|metric - 1|``.

        The direction is looked up in
        ``skfair.comparison._utils.DEFAULT_METRIC_DIRECTION``. The ideal
        point is therefore always top-left: lowest fairness distance,
        highest performance.

        Parameters
        ----------
        fairness_metric : str
            Fairness metric column name (default ``"spd"``).
        performance_metric : str
            Performance metric column name (default ``"accuracy"``).
        datasets : list of str, optional
            Datasets to include. *None* uses all.
        methods : list of str, optional
            Methods to include. *None* uses all.
        classifiers : list of str, optional
            Classifiers to include. *None* uses all.

        Returns
        -------
        tuple of (fig, axes)
        """
        datasets = self._resolve_datasets(datasets)
        methods = self._resolve_methods(methods)
        classifiers = self._resolve_classifiers(classifiers)
        df = self._filter_df(datasets, methods, classifiers)
        return _plot_tradeoff_scatter(df, fairness_metric, performance_metric,
                                      datasets, show_std=show_std, **kw)

    def plot_ranking(self, metrics=None, datasets=None, higher_is_better=None,
                     classifier=None, methods=None, classifiers=None, **kw):
        """Heatmap of method rankings per dataset.

        Parameters
        ----------
        metrics : list of str, optional
            Metrics to rank on. Defaults to all detected metrics.
        datasets : list of str, optional
            Datasets to include. *None* uses all.
        higher_is_better : dict, optional
            Per-metric direction overrides, mapping metric name to one of:

            - ``"higher"`` — higher value is better (performance metrics);
            - ``"zero"``   — closer to 0 is better (difference fairness
              metrics, e.g. ``spd``, ``eod``);
            - ``"one"``    — closer to 1 is better (ratio fairness
              metrics, e.g. ``disparate_impact``,
              ``false_positive_rate_ratio``).

            Defaults to
            ``skfair.comparison._utils.DEFAULT_METRIC_DIRECTION``. The
            parameter name is historical; values are strings, not booleans.
        classifier : None, "average", "best", or a classifier name.
            How to aggregate across classifiers before ranking.
        methods : list of str, optional
            Filter to these methods only.
        classifiers : list of str, optional
            Filter to these classifiers only.

        Returns
        -------
        tuple of (fig, axes)
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
        _classifiers = self._resolve_classifiers(classifiers)
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

        filtered_df = self._filter_df(datasets, methods, _classifiers)

        # Std error bars are available only where a {metric}_std column exists.
        has_std = any(f"{m}_std" in filtered_df.columns for m in metrics)

        def _bar_cell(ds_df, m, ds):
            """One metric-bar cell, wrapped as a std toggle pair when possible."""
            fig, _ = _plot_metric_bar(ds_df, m, [ds])
            plain = _fig_to_img(fig)
            if not (has_std and f"{m}_std" in ds_df.columns):
                return plain
            fig_s, _ = _plot_metric_bar(ds_df, m, [ds], show_std=True)
            return (f'<div class="chart-plain">{plain}</div>'
                    f'<div class="chart-std">{_fig_to_img(fig_s)}</div>')

        # --- Performance charts: {metric: {dataset: img}} ---
        perf_charts = {}
        for m in perf_metrics:
            perf_charts[m] = {}
            for ds in datasets:
                ds_df = filtered_df[filtered_df["dataset"] == ds]
                perf_charts[m][ds] = _bar_cell(ds_df, m, ds)

        # --- Fairness charts: {metric: {dataset: img}} ---
        fair_charts = {}
        for m in fair_metrics:
            fair_charts[m] = {}
            for ds in datasets:
                ds_df = filtered_df[filtered_df["dataset"] == ds]
                fair_charts[m][ds] = _bar_cell(ds_df, m, ds)

        # --- Ranking charts: {dataset: {agg: img}} ---
        rank_charts = {}
        for ds in datasets:
            agg_imgs = {}
            for clf in _classifiers:
                fig, _ = _plot_ranking_heatmap(filtered_df, metrics, [ds],
                                                classifier=clf)
                agg_imgs[clf] = _fig_to_img(fig)
            for agg_mode in ["best", "average"]:
                fig, _ = _plot_ranking_heatmap(filtered_df, metrics, [ds],
                                                classifier=agg_mode)
                label = agg_mode.capitalize()
                agg_imgs[label] = _fig_to_img(fig)
            rank_charts[ds] = agg_imgs

        # --- Tradeoff charts: {fairness_metric: {dataset: img}} ---
        tradeoff_charts = {}
        if fair_metrics and pm:
            for fm in fair_metrics:
                tradeoff_charts[fm] = {}
                for ds in datasets:
                    ds_df = filtered_df[filtered_df["dataset"] == ds]
                    fig, _ = _plot_tradeoff_scatter(ds_df, fm, pm, [ds])
                    plain = _fig_to_img(fig)
                    if has_std and (f"{fm}_std" in ds_df.columns
                                    or f"{pm}_std" in ds_df.columns):
                        fig_s, _ = _plot_tradeoff_scatter(ds_df, fm, pm, [ds],
                                                          show_std=True)
                        tradeoff_charts[fm][ds] = (
                            f'<div class="chart-plain">{plain}</div>'
                            f'<div class="chart-std">{_fig_to_img(fig_s)}</div>')
                    else:
                        tradeoff_charts[fm][ds] = plain

        # --- Tables: {dataset: {agg: df}} ---
        tables_avg = _summary_tables(filtered_df, metrics, datasets, classifier="average")
        tables_best = _summary_tables(filtered_df, metrics, datasets, classifier="best")
        tables = {}
        for ds in datasets:
            tables[ds] = {}
            for clf_name in _classifiers:
                clf_tables = _summary_tables(filtered_df, metrics, datasets, classifier=clf_name)
                if ds in clf_tables:
                    tables[ds][clf_name] = clf_tables[ds]
            if ds in tables_best:
                tables[ds]["Best"] = tables_best[ds]
            if ds in tables_avg:
                tables[ds]["Average"] = tables_avg[ds]

        metadata = {
            "n_datasets": len(datasets),
            "n_methods": len(methods),
            "n_classifiers": len(_classifiers),
            "n_metrics": len(metrics),
        }

        html = render_html_report(
            perf_charts=perf_charts,
            fair_charts=fair_charts,
            rank_charts=rank_charts,
            tradeoff_charts=tradeoff_charts,
            tables=tables,
            metadata=metadata,
            datasets=datasets,
            has_std=has_std,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    def plot_all(self, datasets=None, fairness_metric="spd", methods=None, classifiers=None):
        """Run all plot methods and return a list of (fig, axes).

        Parameters
        ----------
        datasets : list of str, optional
            Datasets to include. *None* uses all.
        fairness_metric : str
            Fairness metric for tradeoff plot (default ``"spd"``).
        methods : list of str, optional
            Methods to include. *None* uses all.
        classifiers : list of str, optional
            Classifiers to include. *None* uses all.

        Returns
        -------
        list of (fig, axes) tuples
        """
        datasets = self._resolve_datasets(datasets)
        fm = fairness_metric if fairness_metric in self.fairness_metrics else self.fairness_metrics[0]
        results = []
        for m in self.performance_metrics:
            results.append(self.plot_metric_bar(
                metric=m, datasets=datasets, methods=methods, classifiers=classifiers))
        for m in self.fairness_metrics:
            results.append(self.plot_metric_bar(
                metric=m, datasets=datasets, methods=methods, classifiers=classifiers))
        if self.fairness_metrics and self.performance_metrics:
            results.append(self.plot_tradeoff(
                fairness_metric=fm,
                performance_metric=self.performance_metrics[0],
                datasets=datasets, methods=methods, classifiers=classifiers))
        results.append(self.plot_ranking(datasets=datasets, methods=methods, classifiers=classifiers))
        return results
