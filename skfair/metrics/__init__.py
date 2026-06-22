"""Fairness and performance metrics for binary classification.

Fairness metrics are organised in counterpart pairs: each base measure exposes
both a *difference* form (suffix ``_difference``, ideal = 0) and a *ratio* form
(suffix ``_ratio``, ideal = 1). Primary names are the full descriptive forms;
short aliases (``spd``, ``di``, ``eod``, ``eor``, ``aod``, ``fpr_diff``,
``ppv_ratio``, ...) are also exported. All metric metadata lives in the single
registry :mod:`skfair.metrics._registry`, which also provides the
:func:`benchmark` helper and the :data:`METRICS` view.
"""

from ._registry import (
    METRICS,
    DEFAULT_BENCHMARK_METRICS,
    MetricSpec,
    REGISTRY,
    benchmark,
    direction,
    is_fairness,
    kind_of,
    resolve,
)
from ._fairness import (
    # independence
    statistical_parity_difference,
    disparate_impact,
    spd,
    di,
    # separation: TPR
    true_positive_rate_difference,
    true_positive_rate_ratio,
    eod,
    tpr_diff,
    eor,
    tpr_ratio,
    equal_opportunity_difference,
    equal_opportunity_ratio,
    # separation: FPR
    false_positive_rate_difference,
    false_positive_rate_ratio,
    fpr_diff,
    fpr_ratio,
    # separation: TNR
    true_negative_rate_difference,
    true_negative_rate_ratio,
    tnr_diff,
    tnr_ratio,
    # separation: FNR
    false_negative_rate_difference,
    false_negative_rate_ratio,
    fnr_diff,
    fnr_ratio,
    # separation: combined odds
    average_odds_difference,
    average_odds_ratio,
    aod,
    aor,
    # sufficiency: PPV
    positive_predictive_value_difference,
    positive_predictive_value_ratio,
    ppv_diff,
    ppv_ratio,
    # sufficiency: NPV
    negative_predictive_value_difference,
    negative_predictive_value_ratio,
    npv_diff,
    npv_ratio,
    # sufficiency: FDR
    false_discovery_rate_difference,
    false_discovery_rate_ratio,
    fdr_diff,
    fdr_ratio,
    # sufficiency: FOR
    false_omission_rate_difference,
    false_omission_rate_ratio,
    for_diff,
    for_ratio,
    # accuracy
    accuracy_difference,
    accuracy_ratio,
    acc_diff,
    acc_ratio,
)
from ._performance import (
    accuracy,
    balanced_accuracy,
    f1_score,
    false_discovery_rate,
    false_negative_rate,
    false_omission_rate,
    false_positive_rate,
    geometric_mean,
    negative_predictive_value,
    positive_predictive_value,
    precision,
    recall,
    true_negative_rate,
    true_positive_rate,
)

__all__ = [
    # ----- Performance -----
    "accuracy",
    "true_positive_rate",
    "false_positive_rate",
    "true_negative_rate",
    "false_negative_rate",
    "balanced_accuracy",
    "geometric_mean",
    "precision",
    "recall",
    "f1_score",
    "positive_predictive_value",
    "negative_predictive_value",
    "false_discovery_rate",
    "false_omission_rate",
    # ----- Fairness: registry / benchmark -----
    "METRICS",
    "REGISTRY",
    "MetricSpec",
    "DEFAULT_BENCHMARK_METRICS",
    "benchmark",
    "resolve",
    "direction",
    "is_fairness",
    "kind_of",
    # ----- Fairness: independence -----
    "statistical_parity_difference",
    "disparate_impact",
    "spd",
    "di",
    # ----- Fairness: separation (TPR) -----
    "true_positive_rate_difference",
    "true_positive_rate_ratio",
    "eod",
    "tpr_diff",
    "eor",
    "tpr_ratio",
    "equal_opportunity_difference",
    "equal_opportunity_ratio",
    # ----- Fairness: separation (FPR) -----
    "false_positive_rate_difference",
    "false_positive_rate_ratio",
    "fpr_diff",
    "fpr_ratio",
    # ----- Fairness: separation (TNR) -----
    "true_negative_rate_difference",
    "true_negative_rate_ratio",
    "tnr_diff",
    "tnr_ratio",
    # ----- Fairness: separation (FNR) -----
    "false_negative_rate_difference",
    "false_negative_rate_ratio",
    "fnr_diff",
    "fnr_ratio",
    # ----- Fairness: separation (combined odds) -----
    "average_odds_difference",
    "average_odds_ratio",
    "aod",
    "aor",
    # ----- Fairness: sufficiency (PPV) -----
    "positive_predictive_value_difference",
    "positive_predictive_value_ratio",
    "ppv_diff",
    "ppv_ratio",
    # ----- Fairness: sufficiency (NPV) -----
    "negative_predictive_value_difference",
    "negative_predictive_value_ratio",
    "npv_diff",
    "npv_ratio",
    # ----- Fairness: sufficiency (FDR) -----
    "false_discovery_rate_difference",
    "false_discovery_rate_ratio",
    "fdr_diff",
    "fdr_ratio",
    # ----- Fairness: sufficiency (FOR) -----
    "false_omission_rate_difference",
    "false_omission_rate_ratio",
    "for_diff",
    "for_ratio",
    # ----- Fairness: accuracy -----
    "accuracy_difference",
    "accuracy_ratio",
    "acc_diff",
    "acc_ratio",
]
