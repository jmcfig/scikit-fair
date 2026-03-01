"""Fairness and performance metrics for binary classification."""

from ._fairness import (
    average_odds_difference,
    disparate_impact,
    equal_opportunity_difference,
    statistical_parity_difference,
    true_negative_rate_difference,
)
from ._performance import (
    accuracy,
    balanced_accuracy,
    false_negative_rate,
    false_positive_rate,
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
    # Fairness
    "disparate_impact",
    "statistical_parity_difference",
    "equal_opportunity_difference",
    "average_odds_difference",
    "true_negative_rate_difference",
]
