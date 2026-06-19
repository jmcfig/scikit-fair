"""Utility functions for comparison report: validation, metric detection, ranking."""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"dataset", "method", "classifier"}

# Maps metric name -> direction.
# Organised in counterpart pairs: every base measure has a difference form
# ("zero", ideal = 0) and a ratio form ("one", ideal = 1). Both canonical
# names and short aliases are listed.
DEFAULT_METRIC_DIRECTION = {
    # Performance metrics (higher is better)
    "accuracy": "higher",
    "balanced_accuracy": "higher",
    "f1": "higher",
    "f1_score": "higher",
    "precision": "higher",
    "recall": "higher",
    "roc_auc": "higher",
    "true_positive_rate": "higher",
    "false_positive_rate": "higher",
    "true_negative_rate": "higher",
    "false_negative_rate": "higher",
    # Fairness — independence
    "spd": "zero",
    "statistical_parity_difference": "zero",
    "disparate_impact": "one",
    "di": "one",
    # Fairness — separation: TPR
    "eod": "zero",
    "true_positive_rate_difference": "zero",
    "equal_opportunity_difference": "zero",
    "true_positive_rate_ratio": "one",
    "equal_opportunity_ratio": "one",
    "eor": "one",
    # Fairness — separation: FPR
    "false_positive_rate_difference": "zero",
    "fpr_diff": "zero",
    "false_positive_rate_ratio": "one",
    "fpr_ratio": "one",
    # Fairness — separation: TNR
    "true_negative_rate_difference": "zero",
    "tnr_diff": "zero",
    "true_negative_rate_ratio": "one",
    "tnr_ratio": "one",
    # Fairness — separation: FNR
    "false_negative_rate_difference": "zero",
    "fnr_diff": "zero",
    "false_negative_rate_ratio": "one",
    "fnr_ratio": "one",
    # Fairness — separation: combined odds (FPR + TPR)
    "aod": "zero",
    "average_odds_difference": "zero",
    "average_odds_ratio": "one",
    "aor": "one",
    # Fairness — sufficiency: PPV
    "positive_predictive_value_difference": "zero",
    "ppv_diff": "zero",
    "predictive_parity_difference": "zero",
    "positive_predictive_value_ratio": "one",
    "ppv_ratio": "one",
    "predictive_parity_ratio": "one",
    # Fairness — sufficiency: NPV
    "negative_predictive_value_difference": "zero",
    "npv_diff": "zero",
    "negative_predictive_value_ratio": "one",
    "npv_ratio": "one",
    # Fairness — sufficiency: FDR
    "false_discovery_rate_difference": "zero",
    "fdr_diff": "zero",
    "false_discovery_rate_ratio": "one",
    "fdr_ratio": "one",
    # Fairness — sufficiency: FOR
    "false_omission_rate_difference": "zero",
    "for_diff": "zero",
    "false_omission_rate_ratio": "one",
    "for_ratio": "one",
    # Fairness — accuracy
    "accuracy_difference": "zero",
    "acc_diff": "zero",
    "accuracy_ratio": "one",
    "acc_ratio": "one",
}

# Known fairness metric names (canonical names + aliases).
_FAIRNESS_METRICS = {
    # independence
    "spd", "statistical_parity_difference", "disparate_impact", "di",
    # separation: TPR
    "eod", "true_positive_rate_difference", "equal_opportunity_difference",
    "tpr_diff", "eor", "true_positive_rate_ratio", "equal_opportunity_ratio",
    "tpr_ratio",
    # separation: FPR
    "false_positive_rate_difference", "fpr_diff",
    "false_positive_rate_ratio", "fpr_ratio",
    # separation: TNR
    "true_negative_rate_difference", "tnr_diff",
    "true_negative_rate_ratio", "tnr_ratio",
    # separation: FNR
    "false_negative_rate_difference", "fnr_diff",
    "false_negative_rate_ratio", "fnr_ratio",
    # separation: combined odds (FPR + TPR)
    "aod", "average_odds_difference", "average_odds_ratio", "aor",
    # sufficiency: PPV
    "positive_predictive_value_difference", "ppv_diff",
    "predictive_parity_difference", "positive_predictive_value_ratio",
    "ppv_ratio", "predictive_parity_ratio",
    # sufficiency: NPV
    "negative_predictive_value_difference", "npv_diff",
    "negative_predictive_value_ratio", "npv_ratio",
    # sufficiency: FDR
    "false_discovery_rate_difference", "fdr_diff",
    "false_discovery_rate_ratio", "fdr_ratio",
    # sufficiency: FOR
    "false_omission_rate_difference", "for_diff",
    "false_omission_rate_ratio", "for_ratio",
    # accuracy
    "accuracy_difference", "acc_diff", "accuracy_ratio", "acc_ratio",
}


def validate_results_df(df):
    """Check that df has required columns and at least one metric column."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not detect_metrics(df):
        raise ValueError("No metric columns found in DataFrame")


def detect_metrics(df):
    """Return list of metric column names.

    A metric column is any column not in the required set
    (dataset, method, classifier) and not ending with ``_std``.
    """
    return [c for c in df.columns
            if c not in REQUIRED_COLUMNS
            and not c.endswith("_std")]


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
            col = metric
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


def _aggregate_by_classifier(sub, metric_cols, classifier):
    """Aggregate a per-dataset subset according to the *classifier* parameter.

    Parameters
    ----------
    sub : DataFrame
        Rows for a single dataset.
    metric_cols : list of str
    classifier : None, "average", "best", or a classifier name.

    Returns
    -------
    DataFrame indexed by ``method`` with one column per metric.
    """
    if classifier is None or classifier == "average":
        return sub.groupby("method")[metric_cols].mean()

    if classifier == "best":
        rows = []
        for method, grp in sub.groupby("method"):
            best_vals = {}
            for m in metric_cols:
                direction = DEFAULT_METRIC_DIRECTION.get(m, "higher")
                col_vals = grp[m].dropna()
                if col_vals.empty:
                    best_vals[m] = np.nan
                elif direction == "higher":
                    best_vals[m] = col_vals.max()
                elif direction == "zero":
                    best_vals[m] = col_vals.iloc[col_vals.abs().argmin()]
                elif direction == "one":
                    best_vals[m] = col_vals.iloc[(col_vals - 1.0).abs().argmin()]
                else:
                    best_vals[m] = col_vals.max()
            rows.append({"method": method, **best_vals})
        return pd.DataFrame(rows).set_index("method")

    # Specific classifier name
    filtered = sub[sub["classifier"] == classifier]
    if filtered.empty:
        available = sorted(sub["classifier"].unique())
        raise ValueError(
            f"Classifier '{classifier}' not found. Available: {available}"
        )
    return filtered.set_index("method")[metric_cols]


def _classifier_title_suffix(classifier):
    """Return a parenthesized suffix for plot/table titles."""
    if classifier is None or classifier == "average":
        return "(averaged over classifiers)"
    if classifier == "best":
        return "(best classifier per metric)"
    return f"({classifier})"
