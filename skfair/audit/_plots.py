"""Shared plotting helpers for the audit module.

All functions return ``(fig, ax)`` so callers can customise further.
Matplotlib-first; seaborn is used only where it adds real value.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bar chart from a Series / single-column DataFrame
# ---------------------------------------------------------------------------

def _plot_bar(
    data: pd.Series,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    color: str = "#4c72b0",
    horizontal: bool = False,
    figsize: tuple = (8, 4),
) -> tuple:
    """Simple bar chart from a pandas Series.

    Parameters
    ----------
    data : pd.Series
        Values to plot; index is used for tick labels.
    title, xlabel, ylabel : str
        Plot labels.
    color : str
        Bar colour.
    horizontal : bool
        If True draw horizontal bars.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        ax.barh(data.index.astype(str), data.values, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.bar(data.index.astype(str), data.values, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Grouped bar chart (two groups side by side)
# ---------------------------------------------------------------------------

def _plot_grouped_bars(
    df: pd.DataFrame,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (9, 5),
) -> tuple:
    """Grouped bar chart.  Each column is a group; rows are categories.

    Parameters
    ----------
    df : pd.DataFrame
        Rows = metric names, columns = group labels, values = metric values.

    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_metrics = len(df)
    n_groups = len(df.columns)
    x = np.arange(n_metrics)
    width = 0.8 / n_groups
    colours = plt.cm.Set2.colors  # noqa: N806

    for i, col in enumerate(df.columns):
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, df[col], width, label=col,
               color=colours[i % len(colours)])

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Fairness metric bars with threshold lines
# ---------------------------------------------------------------------------

def _plot_metric_bars(
    metrics: pd.Series,
    *,
    title: str = "Fairness Metrics",
    figsize: tuple = (9, 5),
) -> tuple:
    """Bar chart for fairness metrics with ideal-value reference lines.

    Ratio metrics (ideal = 1) and difference metrics (ideal = 0) are
    distinguished automatically by name suffix.

    Parameters
    ----------
    metrics : pd.Series
        Metric name -> value.

    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = metrics.index.tolist()
    values = metrics.values.astype(float)

    # Colour by whether value is "good" (close to ideal)
    _ratio_keywords = {"ratio", "impact", "parity", "equality"}
    colours = []
    for name, val in zip(names, values):
        name_lower = name.lower().replace(" ", "_")
        is_ratio = any(k in name_lower for k in _ratio_keywords)
        ideal = 1.0 if is_ratio else 0.0
        dist = abs(val - ideal)
        if dist <= 0.1:
            colours.append("#2ca02c")  # green
        elif dist <= 0.2:
            colours.append("#ff7f0e")  # orange
        else:
            colours.append("#d62728")  # red

    bars = ax.barh(names, values, color=colours)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.axvline(1, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Value")
    ax.set_title(title)

    # value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Radar / spider chart
# ---------------------------------------------------------------------------

def _plot_radar(
    metrics: pd.Series,
    *,
    title: str = "Fairness Radar",
    figsize: tuple = (7, 7),
) -> tuple:
    """Radar (spider) chart for fairness metrics.

    All values are shown on radial axes.  Ratio metrics have an ideal of 1;
    difference metrics have an ideal of 0.

    Parameters
    ----------
    metrics : pd.Series
        Metric name -> value.

    Returns
    -------
    (fig, ax)
    """
    labels = metrics.index.tolist()
    values = metrics.values.astype(float).tolist()

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    # close the polygon
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2, color="#4c72b0")
    ax.fill(angles, values, alpha=0.25, color="#4c72b0")
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_title(title, y=1.08)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Histogram split by group
# ---------------------------------------------------------------------------

def _plot_histogram_by_group(
    values: pd.Series,
    group_labels: np.ndarray,
    *,
    feature_name: str = "",
    group_name: str = "Group",
    bins: int = 20,
    figsize: tuple = (8, 4),
) -> tuple:
    """Overlaid histograms for a feature, split by binary group.

    Parameters
    ----------
    values : pd.Series
        Feature values.
    group_labels : np.ndarray
        Binary group indicator (0/1) aligned with *values*.
    feature_name : str
        Label for the x-axis.
    group_name : str
        Label used in the legend prefix.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    group_labels = np.asarray(group_labels)

    unique_groups = np.unique(group_labels)
    colours = plt.cm.Set2.colors  # noqa: N806

    for i, g in enumerate(unique_groups):
        mask = group_labels == g
        ax.hist(values[mask], bins=bins, alpha=0.55,
                label=f"{group_name}={g}", color=colours[i % len(colours)])

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {feature_name} by {group_name}")
    ax.legend()
    fig.tight_layout()
    return fig, ax
