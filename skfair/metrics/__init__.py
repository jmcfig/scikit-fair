"""Fairness and performance metrics for binary classification.

Fairness metrics are organised in counterpart pairs: each base measure
exposes both a *difference* form (ideal = 0) and a *ratio/parity* form
(ideal = 1).
"""

from ._fairness import (
    accuracy_difference,
    accuracy_parity,
    average_odds_difference,
    average_odds_ratio,
    disparate_impact,
    equal_opportunity_difference,
    equal_opportunity_ratio,
    false_negative_rate_difference,
    false_negative_rate_parity,
    false_positive_rate_difference,
    false_positive_rate_parity,
    predictive_equality,
    statistical_parity_difference,
    true_negative_rate_difference,
    true_negative_rate_parity,
)
from ._performance import (
    accuracy,
    balanced_accuracy,
    f1_score,
    false_negative_rate,
    false_positive_rate,
    geometric_mean,
    precision,
    recall,
    true_negative_rate,
    true_positive_rate,
)

__all__ = [
    # Performance
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
    # Fairness — Positive prediction rate
    "statistical_parity_difference",
    "disparate_impact",
    # Fairness — TPR
    "equal_opportunity_difference",
    "equal_opportunity_ratio",
    # Fairness — FPR
    "false_positive_rate_difference",
    "predictive_equality",
    "false_positive_rate_parity",  # alias of predictive_equality
    # Fairness — TNR
    "true_negative_rate_difference",
    "true_negative_rate_parity",
    # Fairness — FNR
    "false_negative_rate_difference",
    "false_negative_rate_parity",
    # Fairness — Accuracy
    "accuracy_difference",
    "accuracy_parity",
    # Fairness — Combined (FPR + TPR)
    "average_odds_difference",
    "average_odds_ratio",
]
