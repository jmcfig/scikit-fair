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


def _plot_metric_bar(df, metric, datasets, reference_line="auto", figsize=None,
                     show_std=False):
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
    show_std : bool, default False
        Draw ``{metric}_std`` as symmetric error bars on each bar (only where
        the std column is present and non-zero).
    """
    col_name = metric
    std_col = f"{metric}_std"
    has_std = show_std and std_col in df.columns
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
        # Explicit (appearance-order) orderings so error bars can be mapped
        # back onto the dodged bar positions deterministically.
        order = list(dict.fromkeys(sub["method"]))
        hue_order = list(dict.fromkeys(sub["classifier"]))
        sns.barplot(data=sub, x="method", y=col_name, hue="classifier",
                    order=order, hue_order=hue_order, ax=ax, errorbar=None)
        if has_std:
            _overlay_std_bars(ax, sub, col_name, std_col, order, hue_order)
        if reference_line is not None:
            ax.axhline(y=reference_line, color="black", linestyle="-", linewidth=0.8)
        # Tighten y-axis (widen to include error bars when shown)
        if has_std:
            err = sub[std_col].fillna(0.0)
            ymin = (sub[col_name] - err).min()
            ymax = (sub[col_name] + err).max()
        else:
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


def _overlay_std_bars(ax, sub, col_name, std_col, order, hue_order):
    """Overlay symmetric error bars onto a seaborn dodged barplot.

    Bar centers are reconstructed from seaborn's dodge layout (slot width 0.8,
    one sub-bar per hue level) so the std is matched to the right bar without
    relying on ``ax.patches`` ordering.
    """
    bar_w = 0.8
    n_hue = len(hue_order)
    lookup = {
        (r["method"], r["classifier"]): (r[col_name], r[std_col])
        for _, r in sub.iterrows()
    }
    for j, method in enumerate(order):
        for i, clf in enumerate(hue_order):
            if (method, clf) not in lookup:
                continue
            val, err = lookup[(method, clf)]
            if pd.isna(err) or err <= 0:
                continue
            x = j - bar_w / 2 + (i + 0.5) * bar_w / n_hue
            ax.errorbar(x, val, yerr=err, fmt="none", ecolor="#333333",
                        capsize=3, elinewidth=1, zorder=10)


# ---------------------------------------------------------------------------
# Tradeoff scatter
# ---------------------------------------------------------------------------

def _plot_tradeoff_scatter(df, fairness_metric, performance_metric, datasets,
                           figsize=None, show_std=False):
    """Scatter of fairness distance vs performance, faceted by dataset.

    The x-axis is direction-aware. The metric direction is looked up in
    ``DEFAULT_METRIC_DIRECTION`` (default ``"zero"`` for unknown metrics):

    - direction ``"zero"`` (difference metrics, ideal = 0) plots
      ``|metric|`` with x-label ``|{metric}|  (lower = fairer)``;
    - direction ``"one"`` (ratio metrics, ideal = 1) plots
      ``|metric - 1|`` with x-label ``|{metric} - 1|  (lower = fairer)``.

    In both cases the ideal point sits in the top-left corner: low x
    (fairest) and high y (best performance). Hue = method,
    style = classifier.
    """
    f_col = fairness_metric
    p_col = performance_metric

    direction = DEFAULT_METRIC_DIRECTION.get(fairness_metric, "zero")
    if direction == "one":
        xlabel = f"|{fairness_metric} - 1|  (lower = fairer)"
    else:
        xlabel = f"|{fairness_metric}|  (lower = fairer)"

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
        if direction == "one":
            sub["_abs_fairness"] = (sub[f_col] - 1.0).abs()
        else:
            sub["_abs_fairness"] = sub[f_col].abs()
        if show_std:
            xerr = sub[f"{f_col}_std"] if f"{f_col}_std" in sub.columns else None
            yerr = sub[f"{p_col}_std"] if f"{p_col}_std" in sub.columns else None
            if xerr is not None or yerr is not None:
                ax.errorbar(
                    sub["_abs_fairness"], sub[p_col], xerr=xerr, yerr=yerr,
                    fmt="none", ecolor="#999999", elinewidth=1, alpha=0.5, zorder=0,
                )
        sns.scatterplot(
            data=sub, x="_abs_fairness", y=p_col,
            hue="method", style="classifier",
            ax=ax, alpha=0.85, s=90,
        )
        ax.set_title(ds, fontsize=12)
        ax.set_xlabel(xlabel)
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

        # Highlight the AVERAGE column
        if "avg" in heatmap_data.columns:
            avg_col_idx = heatmap_data.columns.get_loc("avg")
            # Separator line before the avg column
            ax.axvline(avg_col_idx, color="black", linewidth=2)
            # Bold "AVERAGE" label
            xticks = ax.get_xticklabels()
            for label in xticks:
                if label.get_text() == "avg":
                    label.set_text("AVERAGE")
                    label.set_fontweight("bold")
            ax.set_xticklabels(xticks)
            # Bold cell annotations in the avg column
            avg_x = avg_col_idx + 0.5
            for txt in ax.texts:
                if abs(txt.get_position()[0] - avg_x) < 0.1:
                    txt.set_fontweight("bold")

    for idx in range(len(datasets), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    suffix = _classifier_title_suffix(classifier)
    fig.suptitle(f"Method Rankings (1 = best) {suffix}", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, axes
