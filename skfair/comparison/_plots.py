"""Private plot/table functions for ComparisonReport."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from ._utils import (
    classify_metric,
    compute_rankings,
    DEFAULT_METRIC_DIRECTION,
    _aggregate_by_classifier,
    _classifier_title_suffix,
)

MAX_COLS = 4

#: Bar colours by distance from the metric's ideal, matching the audit design
#: in :func:`skfair.audit._plots._plot_metric_bars`.
_FAIR_COLOUR = "#2ca02c"     # green: within fair_threshold of ideal
_WARN_COLOUR = "#ff7f0e"     # orange: within warning_threshold
_POOR_COLOUR = "#d62728"     # red: beyond it


def _ideal_of(metric):
    """Ideal value for *metric*, or None when it has none (e.g. accuracy)."""
    direction = DEFAULT_METRIC_DIRECTION.get(metric, "higher")
    if direction == "zero":
        return 0.0
    if direction == "one":
        return 1.0
    return None


def _use_ranked_layout(df, metric):
    """True when the ranked horizontal layout applies.

    It needs a single classifier -- with more than one, colour encodes the
    classifier and cannot also encode distance from ideal -- and a metric with
    an ideal value to measure that distance from.
    """
    if "classifier" in df.columns and df["classifier"].nunique() > 1:
        return False
    return _ideal_of(metric) is not None


def _plot_metric_bar(df, metric, datasets, reference_line="auto", figsize=None,
                     show_std=False, xtick_size=None):
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
    xtick_size : float, optional
        Font size of the x-axis (method) tick labels; None uses the rc
        default. Useful when many methods make the labels crowd.

    Notes
    -----
    With a single classifier and a metric that has an ideal value, this
    delegates to :func:`_plot_metric_bars_ranked` -- see there for why.
    """
    if _use_ranked_layout(df, metric):
        return _plot_metric_bars_ranked(
            df, metric, datasets, reference_line=reference_line,
            figsize=figsize, show_std=show_std, tick_size=xtick_size,
        )

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
        ax.set_title(ds)
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " ").title() if idx % ncols == 0 else "")
        _style_xaxis(ax, size=xtick_size)
        legend = ax.get_legend()
        if legend:
            if idx == len(datasets) - 1:
                legend.set_bbox_to_anchor((1.02, 1))
                legend.set_loc("upper left")
                legend.set_title("Classifier")
            else:
                legend.remove()

    for idx in range(len(datasets), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle(f"{metric.replace('_', ' ').title()} by Method", y=1.02)
    fig.tight_layout()
    return fig, axes


def _plot_metric_bars_ranked(df, metric, datasets, reference_line="auto",
                             figsize=None, show_std=False, tick_size=None,
                             fair_threshold=0.1, warning_threshold=0.2):
    """Horizontal bars ranked by distance from ideal, one panel per dataset.

    The single-classifier layout. ``_plot_metric_bar`` spends its colour
    channel on the classifier, which is wasted when there is only one; here
    colour carries distance from the metric's ideal instead -- green within
    *fair_threshold*, orange within *warning_threshold*, red beyond it -- the
    same encoding as :func:`skfair.audit._plots._plot_metric_bars`.

    Methods are ordered by that distance with the closest to ideal on top.
    Values keep their sign, so a method that overshoots into the opposite
    direction is visibly distinct from one that falls short by as much.
    ``Baseline`` is bolded, since every other bar is read against it.

    Parameters
    ----------
    df : DataFrame
    metric : str
    datasets : list of str
    reference_line : float, "auto", or None
        Dashed line marking the ideal. "auto" derives it from the metric.
    figsize : tuple, optional
        Defaults to a height that grows with the number of methods.
    show_std : bool, default False
        Draw ``{metric}_std`` as symmetric horizontal error bars.
    tick_size : float, optional
        Font size of the method tick labels.
    fair_threshold, warning_threshold : float
        Distance from ideal for the green and orange bands.

    Returns
    -------
    (fig, axes)
    """
    std_col = f"{metric}_std"
    has_std = show_std and std_col in df.columns
    ideal = _ideal_of(metric)

    if reference_line == "auto":
        reference_line = ideal

    ncols = min(len(datasets), MAX_COLS)
    nrows = int(np.ceil(len(datasets) / ncols))

    if figsize is None:
        n_methods = max(df["method"].nunique(), 1)
        figsize = (7.0 * ncols, max(2.4, 0.42 * n_methods + 1.4) * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    flat_axes = axes.ravel()
    for idx, ds in enumerate(datasets):
        ax = flat_axes[idx]
        sub = df[(df["dataset"] == ds) & df[metric].notna()].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        # closest to ideal on top: barh draws the first row at the bottom
        sub["_dist"] = (sub[metric] - ideal).abs()
        sub = sub.sort_values("_dist", ascending=False)

        colours = [
            _FAIR_COLOUR if d <= fair_threshold
            else _WARN_COLOUR if d <= warning_threshold
            else _POOR_COLOUR
            for d in sub["_dist"]
        ]
        bars = ax.barh(sub["method"], sub[metric], color=colours)

        if has_std:
            err = sub[std_col].fillna(0.0)
            ax.errorbar(sub[metric], range(len(sub)), xerr=err, fmt="none",
                        ecolor="#333333", capsize=3, elinewidth=1, zorder=10)

        if reference_line is not None:
            ax.axvline(reference_line, color="grey", linewidth=0.8, linestyle="--")

        # Bars grow from zero, so the axis has to hold zero, the ideal and the
        # error bars; the right side gets extra room for the value labels.
        lo = min(sub[metric].min(), ideal, 0.0)
        hi = max(sub[metric].max(), ideal, 0.0)
        if has_std:
            lo = min(lo, (sub[metric] - err).min())
            hi = max(hi, (sub[metric] + err).max())
        pad = (hi - lo) * 0.12 or 0.05
        ax.set_xlim(lo - pad, hi + pad * 3)

        # value labels: at the bar tip for positive values, just clear of the
        # baseline for negative ones, so they never sit on top of a bar --
        # pushed out past the whisker when error bars are drawn
        offset = (hi - lo + 2 * pad) * 0.015
        tips = sub[metric] + err if has_std else sub[metric]
        for bar, val, tip in zip(bars, sub[metric], tips):
            ax.text(max(tip, 0.0) + offset,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left")

        for label in ax.get_yticklabels():
            if tick_size is not None:
                label.set_fontsize(tick_size)
            if label.get_text() == "Baseline":
                label.set_fontweight("bold")

        ax.set_title(ds)
        ax.set_xlabel(f"{metric.replace('_', ' ').title()}  ({ideal:g} = ideal)")
        ax.set_ylabel("")

    for idx in range(len(datasets), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle(f"{metric.replace('_', ' ').title()} by Method", y=1.02)
    fig.tight_layout()
    return fig, axes


def _style_xaxis(ax, size=None):
    """Rotate x-tick labels for readability."""
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        if size is not None:
            label.set_fontsize(size)


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
        ax.set_title(ds)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(performance_metric.replace("_", " ").title() if idx % ncols == 0 else "")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                  title="method / clf")

    for idx in range(len(datasets), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle(
        f"Performance vs Fairness Trade-off (top-left = ideal)",
        y=1.02,
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
                          classifier=None, figsize=None, annot_size=11):
    """Annotated heatmap of method rankings per dataset.

    Green=rank 1, red=worst rank. ``annot_size`` sets the font size of the
    rank numbers inside the cells.
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
            annot_kws={"size": annot_size},
        )
        ax.set_title(ds)
        ax.set_ylabel("")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

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
    fig.suptitle(f"Method Rankings (1 = best) {suffix}", y=1.02)
    fig.tight_layout()
    return fig, axes
