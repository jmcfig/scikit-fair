"""Private plot/table functions for ComparisonReport."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ._utils import (
    classify_metric,
    compute_rankings,
    DEFAULT_METRIC_DIRECTION,
    _aggregate_by_classifier,
    _classifier_title_suffix,
)

MAX_COLS = 4


def _plot_metric_bar(df, metric, datasets, reference_line="auto", figsize=None):
    """Grouped bar chart: x=method, hue=classifier, one panel per dataset.

    Parameters
    ----------
    df : DataFrame
    metric : str
        Single metric column name.
    datasets : list of str
    reference_line : float, "auto", or None
        "auto" derives from metric direction (1.0 for "one", 0.0 for "zero",
        None for "higher"). None disables the line.
    figsize : tuple, optional
    """
    col_name = metric
    ncols = min(len(datasets), MAX_COLS)
    nrows = int(np.ceil(len(datasets) / ncols))

    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # Resolve reference line
    if reference_line == "auto":
        direction = DEFAULT_METRIC_DIRECTION.get(metric, "higher")
        reference_line = 1.0 if direction == "one" else (0.0 if direction == "zero" else None)

    flat_axes = axes.ravel()
    for idx, ds in enumerate(datasets):
        ax = flat_axes[idx]
        sub = df[(df["dataset"] == ds) & df[col_name].notna()].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sns.barplot(data=sub, x="method", y=col_name, hue="classifier",
                    ax=ax, errorbar=None)
        if reference_line is not None:
            ax.axhline(y=reference_line, color="black", linestyle="-", linewidth=0.8)
        # Tighten y-axis
        ymin = sub[col_name].min()
        ymax = sub[col_name].max()
        margin = (ymax - ymin) * 0.15 if ymax > ymin else 0.01
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_title(ds, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " ").title() if idx % ncols == 0 else "")
        _style_xaxis(ax)
        legend = ax.get_legend()
        if legend:
            if idx == len(datasets) - 1:
                legend.set_bbox_to_anchor((1.02, 1))
                legend.set_loc("upper left")
                for text in legend.get_texts():
                    text.set_fontsize(7)
                legend.set_title("Classifier", prop={"size": 8})
            else:
                legend.remove()

    for idx in range(len(datasets), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle(f"{metric.replace('_', ' ').title()} by Method",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, axes


def _style_xaxis(ax):
    """Rotate x-tick labels for readability."""
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_fontsize(7)


# ---------------------------------------------------------------------------
# Tradeoff scatter
# ---------------------------------------------------------------------------

def _plot_tradeoff_scatter(df, fairness_metric, performance_metric, datasets, figsize=None):
    """Scatter: x=|fairness|, y=performance, hue=method, style=classifier."""
    f_col = fairness_metric
    p_col = performance_metric

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

def _summary_tables(df, metrics, datasets, classifier=None):
    """Return {dataset: pivot_df} with methods as rows, metrics as columns.

    Parameters
    ----------
    classifier : None, "average", "best", or a classifier name.
    """
    metric_cols = [m for m in metrics if m in df.columns]
    result = {}
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        pivot = _aggregate_by_classifier(sub, metric_cols, classifier).round(4)
        result[ds] = pivot
    return result


# ---------------------------------------------------------------------------
# 6. Ranking heatmap
# ---------------------------------------------------------------------------

def _plot_ranking_heatmap(df, metrics, datasets, higher_is_better=None,
                          classifier=None, figsize=None):
    """Annotated heatmap of method rankings per dataset.

    Green=rank 1, red=worst rank.
    """
    metric_cols = [m for m in metrics if m in df.columns]
    # Aggregate according to classifier mode
    agg_parts = []
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        agg = _aggregate_by_classifier(sub, metric_cols, classifier).reset_index()
        agg.insert(0, "dataset", ds)
        agg_parts.append(agg)
    agg_df = pd.concat(agg_parts, ignore_index=True)

    rankings = compute_rankings(agg_df, metrics, higher_is_better)

    ncols = min(len(datasets), MAX_COLS)
    nrows = int(np.ceil(len(datasets) / ncols))

    # Compute rank columns early so we can size the figure
    rank_cols = [c for c in rankings.columns if c.endswith("_rank") and c != "avg_rank"]
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

    suffix = _classifier_title_suffix(classifier)
    fig.suptitle(f"Method Rankings (1 = best) {suffix}", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, axes
