"""Utility functions for comparison report: validation, metric detection, ranking."""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"dataset", "method", "classifier"}

# Maps metric name -> whether higher is better (True) or closer to 0/1 is better
DEFAULT_METRIC_DIRECTION = {
    # Performance metrics (higher is better)
    "accuracy": "higher",
    "balanced_accuracy": "higher",
    "f1": "higher",
    "precision": "higher",
    "recall": "higher",
    "roc_auc": "higher",
    # Fairness metrics — difference-based (closer to 0 is better)
    "spd": "zero",
    "eod": "zero",
    "aod": "zero",
    "statistical_parity_difference": "zero",
    "equal_opportunity_difference": "zero",
    "average_odds_difference": "zero",
    # Fairness metrics — ratio-based (closer to 1 is better)
    "disparate_impact": "one",
}

# Known fairness metric names
_FAIRNESS_METRICS = {
    "spd", "eod", "aod",
    "statistical_parity_difference",
    "equal_opportunity_difference",
    "average_odds_difference",
    "disparate_impact",
}


def validate_results_df(df):
    """Check that df has required columns and at least one *_mean column."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    if not mean_cols:
        raise ValueError("No *_mean metric columns found in DataFrame")


def detect_metrics(df):
    """Return list of base metric names from *_mean columns."""
    return [c.removesuffix("_mean") for c in df.columns if c.endswith("_mean")]


def classify_metric(name):
    """Return 'fairness' or 'performance' for a metric name."""
    if name in _FAIRNESS_METRICS:
        return "fairness"
    return "performance"


def _rank_values(values, direction):
    """Rank values (1=best). Lower rank is better."""
    if direction == "higher":
        # Higher is better -> rank descending
        return pd.Series(values).rank(ascending=False, method="min")
    elif direction == "zero":
        # Closer to 0 is better
        return pd.Series(np.abs(values)).rank(ascending=True, method="min")
    elif direction == "one":
        # Closer to 1 is better
        return pd.Series(np.abs(np.array(values) - 1.0)).rank(ascending=True, method="min")
    else:
        # Default: higher is better
        return pd.Series(values).rank(ascending=False, method="min")


def compute_rankings(df, metrics, higher_is_better=None):
    """Rank methods per dataset/metric.

    Parameters
    ----------
    df : DataFrame
        Results DataFrame (averaged over classifiers).
    metrics : list of str
        Metric base names to rank.
    higher_is_better : dict or None
        Override directions. Keys are metric names, values are direction
        strings: 'higher', 'zero', or 'one'. Falls back to DEFAULT_METRIC_DIRECTION.

    Returns
    -------
    DataFrame with columns: dataset, method, {metric}_rank for each metric,
    plus an 'avg_rank' column.
    """
    if higher_is_better is None:
        higher_is_better = {}

    all_ranks = []
    for dataset_name, grp in df.groupby("dataset"):
        row = {"dataset": dataset_name, "method": grp["method"].values}
        for metric in metrics:
            col = f"{metric}_mean"
            if col not in grp.columns:
                continue
            direction = higher_is_better.get(metric, DEFAULT_METRIC_DIRECTION.get(metric, "higher"))
            ranks = _rank_values(grp[col].values, direction)
            row[f"{metric}_rank"] = ranks.values

        # Build per-dataset DataFrame
        n = len(grp)
        rank_df = pd.DataFrame({"dataset": [dataset_name] * n, "method": grp["method"].values})
        for metric in metrics:
            key = f"{metric}_rank"
            if key in row:
                rank_df[key] = row[key]

        rank_cols = [c for c in rank_df.columns if c.endswith("_rank")]
        if rank_cols:
            rank_df["avg_rank"] = rank_df[rank_cols].mean(axis=1)
        all_ranks.append(rank_df)

    if not all_ranks:
        return pd.DataFrame()
    return pd.concat(all_ranks, ignore_index=True)
