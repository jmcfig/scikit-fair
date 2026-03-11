"""Private plot/table functions for ComparisonReport."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ._utils import classify_metric, compute_rankings, DEFAULT_METRIC_DIRECTION

MAX_COLS = 4


def _make_facet_grid(datasets, nrows=1, figsize_per_panel=(5, 4), sharey=True):
    """Create a subplot grid, always returning a 2D axes array.

    Parameters
    ----------
    datasets : list of str
    nrows : int
        Number of rows in the grid.
    figsize_per_panel : tuple
        (width, height) per panel.
    sharey : bool or str

    Returns
    -------
    fig, axes : matplotlib Figure and 2D ndarray of Axes
    """
    n = len(datasets)
    ncols = min(n, MAX_COLS)
    nrows_actual = max(nrows, int(np.ceil(n / ncols)))

    figw = figsize_per_panel[0] * ncols
    figh = figsize_per_panel[1] * nrows_actual

    fig, axes = plt.subplots(
        nrows_actual, ncols, figsize=(figw, figh),
        sharey=sharey, squeeze=False,
    )

    # Hide unused axes
    total_panels = nrows_actual * ncols
    for idx in range(n, total_panels):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    return fig, axes


def _style_xaxis(ax):
    """Rotate x-tick labels for readability."""
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_fontsize(7)


# ---------------------------------------------------------------------------
# 1. Performance bars
# ---------------------------------------------------------------------------

def _plot_performance_bars(df, metrics, datasets, figsize=None):
    """Grid of grouped bar charts: rows=metrics, cols=datasets.

    Bars grouped by method, hue=classifier. Shared y per metric row.
    """
    nrows = len(metrics)
    ncols = min(len(datasets), MAX_COLS)

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for row_idx, metric in enumerate(metrics):
        col_name = f"{metric}_mean"
        for col_idx, ds in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            sub = df[(df["dataset"] == ds) & df[col_name].notna()].copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            sns.barplot(
                data=sub, x="method", y=col_name, hue="classifier",
                ax=ax, errorbar=None,
            )
            # Tighten y-axis to actual data range
            ymin = sub[col_name].min()
            ymax = sub[col_name].max()
            margin = (ymax - ymin) * 0.15 if ymax > ymin else 0.01
            ax.set_ylim(ymin - margin, ymax + margin)
            ax.set_title(ds if row_idx == 0 else "", fontsize=12)
            ax.set_ylabel(metric.replace("_", " ").title() if col_idx == 0 else "")
            ax.set_xlabel("")
            _style_xaxis(ax)
            # Only keep legend on the last column
            if col_idx < ncols - 1:
                legend = ax.get_legend()
                if legend:
                    legend.remove()

    # Hide extra columns if datasets < MAX_COLS
    for col_idx in range(len(datasets), ncols):
        for row_idx in range(nrows):
            axes[row_idx, col_idx].set_visible(False)

    fig.suptitle("Performance Metrics by Method", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 2. Fairness averaged
# ---------------------------------------------------------------------------

def _plot_fairness_averaged(df, metric, datasets, thresholds=None, figsize=None):
    """Bars averaged over classifiers, with optional threshold reference lines.

    Parameters
    ----------
    metric : str
        Base metric name (e.g. 'disparate_impact').
    thresholds : dict or None
        {label: y_value} for reference lines.
    """
    col_name = f"{metric}_mean"
    ncols = min(len(datasets), MAX_COLS)
    nrows = int(np.ceil(len(datasets) / ncols))

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    if thresholds is None:
        direction = DEFAULT_METRIC_DIRECTION.get(metric, "higher")
        if direction == "one":
            thresholds = {"80% rule (0.8)": 0.8, "Perfect (1.0)": 1.0}
        elif direction == "zero":
            thresholds = {"Perfect (0.0)": 0.0}

    flat_axes = axes.ravel()
    for idx, ds in enumerate(datasets):
        ax = flat_axes[idx]
        sub = df[(df["dataset"] == ds) & df[col_name].notna()].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        agg = sub.groupby("method")[col_name].mean().reset_index()
        sns.barplot(data=agg, x="method", y=col_name, ax=ax,
                    color="steelblue", errorbar=None)
        if thresholds:
            colors = ["orange", "green", "red", "purple"]
            for i, (label, yval) in enumerate(thresholds.items()):
                ax.axhline(y=yval, color=colors[i % len(colors)],
                           linestyle="--", linewidth=1.5, label=label)
            ax.legend(fontsize=8)
        # Tighten y-axis to actual data range
        ymin = agg[col_name].min()
        ymax = agg[col_name].max()
        margin = (ymax - ymin) * 0.15 if ymax > ymin else 0.01
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_title(ds, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " ").title() if idx % ncols == 0 else "")
        _style_xaxis(ax)

    # Hide unused
    for idx in range(len(datasets), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    direction = DEFAULT_METRIC_DIRECTION.get(metric, "higher")
    if direction == "one":
        direction_hint = " (closer to 1 = fairer)"
    elif direction == "zero":
        direction_hint = " (closer to 0 = fairer)"
    else:
        direction_hint = ""

    fig.suptitle(
        f"{metric.replace('_', ' ').title()} by Method (averaged over classifiers){direction_hint}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 3. Fairness detailed
# ---------------------------------------------------------------------------

def _plot_fairness_detailed(df, metric, datasets, reference_line=None, figsize=None):
    """Grouped bars by method, hue=classifier. One panel per dataset."""
    col_name = f"{metric}_mean"
    ncols = min(len(datasets), MAX_COLS)
    nrows = int(np.ceil(len(datasets) / ncols))

    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    if reference_line is None:
        direction = DEFAULT_METRIC_DIRECTION.get(metric, "higher")
        reference_line = 1.0 if direction == "one" else 0.0

    flat_axes = axes.ravel()
    for idx, ds in enumerate(datasets):
        ax = flat_axes[idx]
        sub = df[(df["dataset"] == ds) & df[col_name].notna()].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sns.barplot(
            data=sub, x="method", y=col_name, hue="classifier",
            ax=ax, errorbar=None,
        )
        ax.axhline(y=reference_line, color="black", linestyle="-", linewidth=0.8)
        # Tighten y-axis to actual data range
        ymin = sub[col_name].min()
        ymax = sub[col_name].max()
        margin = (ymax - ymin) * 0.15 if ymax > ymin else 0.01
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_title(ds, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " ").title() if idx % ncols == 0 else "")
        _style_xaxis(ax)
        if idx < len(datasets) - 1:
            legend = ax.get_legend()
            if legend:
                legend.remove()

    for idx in range(len(datasets), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    direction = DEFAULT_METRIC_DIRECTION.get(metric, "higher")
    if direction == "one":
        direction_hint = " (closer to 1 = fairer)"
    elif direction == "zero":
        direction_hint = " (closer to 0 = fairer)"
    else:
        direction_hint = ""

    fig.suptitle(
        f"{metric.replace('_', ' ').title()} by Method (per classifier){direction_hint}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 4. Tradeoff scatter
# ---------------------------------------------------------------------------

def _plot_tradeoff_scatter(df, fairness_metric, performance_metric, datasets, figsize=None):
    """Scatter: x=|fairness|, y=performance, hue=method, style=classifier."""
    f_col = f"{fairness_metric}_mean"
    p_col = f"{performance_metric}_mean"

    ncols = min(len(datasets), MAX_COLS)
    nrows = int(np.ceil(len(datasets) / ncols))

    if figsize is None:
        figsize = (7 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    flat_axes = axes.ravel()
    for idx, ds in enumerate(datasets):
        ax = flat_axes[idx]
        sub = df[(df["dataset"] == ds) & df[f_col].notna() & df[p_col].notna()].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub["_abs_fairness"] = sub[f_col].abs()
        sns.scatterplot(
            data=sub, x="_abs_fairness", y=p_col,
            hue="method", style="classifier",
            ax=ax, alpha=0.85, s=90,
        )
        ax.set_title(ds, fontsize=12)
        ax.set_xlabel(f"|{fairness_metric}|  (lower = fairer)")
        ax.set_ylabel(performance_metric.replace("_", " ").title() if idx % ncols == 0 else "")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
                  title="method / clf")

    for idx in range(len(datasets), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle(
        f"Performance vs Fairness Trade-off (top-left = ideal)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 5. Summary tables
# ---------------------------------------------------------------------------

def _summary_tables(df, metrics, datasets):
    """Return {dataset: pivot_df} with methods as rows, metrics as columns.

    Values are averaged over classifiers.
    """
    mean_cols = [f"{m}_mean" for m in metrics if f"{m}_mean" in df.columns]
    result = {}
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        pivot = sub.groupby("method")[mean_cols].mean().round(4)
        pivot.columns = [c.removesuffix("_mean") for c in pivot.columns]
        result[ds] = pivot
    return result


# ---------------------------------------------------------------------------
# 6. Ranking heatmap
# ---------------------------------------------------------------------------

def _plot_ranking_heatmap(df, metrics, datasets, higher_is_better=None, figsize=None):
    """Annotated heatmap of method rankings per dataset.

    Green=rank 1, red=worst rank.
    """
    # Average over classifiers first
    mean_cols = [f"{m}_mean" for m in metrics if f"{m}_mean" in df.columns]
    agg_df = (
        df.groupby(["dataset", "method"])[mean_cols]
        .mean()
        .reset_index()
    )

    rankings = compute_rankings(agg_df, metrics, higher_is_better)

    ncols = min(len(datasets), MAX_COLS)
    nrows = int(np.ceil(len(datasets) / ncols))

    # Compute rank columns early so we can size the figure
    rank_cols = [c for c in rankings.columns if c.endswith("_rank")]
    n_methods = rankings["method"].nunique()

    if figsize is None:
        figsize = (max(8, len(rank_cols) * 1.5) * ncols, max(6, n_methods * 0.6) * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    flat_axes = axes.ravel()
    for idx, ds in enumerate(datasets):
        ax = flat_axes[idx]
        sub = rankings[rankings["dataset"] == ds].set_index("method")
        display_cols = rank_cols + (["avg_rank"] if "avg_rank" in sub.columns else [])
        heatmap_data = sub[display_cols]
        if heatmap_data.empty:
            ax.set_visible(False)
            continue
        # Rename columns for readability
        heatmap_data = heatmap_data.copy()
        heatmap_data.columns = [c.removesuffix("_rank") for c in heatmap_data.columns]

        max_rank = heatmap_data.max().max()
        sns.heatmap(
            heatmap_data, annot=True, fmt=".1f", ax=ax,
            cmap="RdYlGn_r", vmin=1, vmax=max_rank,
            linewidths=0.8, cbar=False,
            annot_kws={"size": 10},
        )
        ax.set_title(ds, fontsize=12)
        ax.set_ylabel("")

    for idx in range(len(datasets), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle("Method Rankings (1 = best)", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, axes
