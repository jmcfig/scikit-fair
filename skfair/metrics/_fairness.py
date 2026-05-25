"""Group fairness metrics for binary classification.

Metrics are organised in **counterpart pairs**: for every base measure
(positive prediction rate, TPR, FPR, TNR, FNR, accuracy, average odds) the
file exposes a *difference* form (ideal = 0) immediately followed by its
*ratio* form (ideal = 1). Fairness functions import the corresponding
performance metrics from ``_performance.py`` so that formulas read like
their mathematical definitions.
"""

import numpy as np

from ._performance import (
    accuracy,
    false_negative_rate,
    false_positive_rate,
    true_negative_rate,
    true_positive_rate,
)

__all__ = [
    # Positive prediction rate
    "statistical_parity_difference",
    "disparate_impact",
    # TPR
    "equal_opportunity_difference",
    "equal_opportunity_ratio",
    # FPR
    "false_positive_rate_difference",
    "predictive_equality",
    "false_positive_rate_parity",  # alias of predictive_equality
    # TNR
    "true_negative_rate_difference",
    "true_negative_rate_parity",
    # FNR
    "false_negative_rate_difference",
    "false_negative_rate_parity",
    # Accuracy
    "accuracy_difference",
    "accuracy_parity",
    # Combined (FPR + TPR)
    "average_odds_difference",
    "average_odds_ratio",
]


def _split_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> tuple:
    """Split y_true and y_pred into unprivileged and privileged subsets.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    tuple of ((y_true_unpriv, y_pred_unpriv), (y_true_priv, y_pred_priv))
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_attr = np.asarray(sensitive_attr)

    if not (len(y_true) == len(y_pred) == len(sensitive_attr)):
        raise ValueError("y_true, y_pred, and sensitive_attr must have the same length.")

    mask_priv = sensitive_attr == 1
    mask_unpriv = sensitive_attr == 0

    if not mask_priv.any():
        raise ValueError("No privileged samples (sensitive_attr == 1) found.")
    if not mask_unpriv.any():
        raise ValueError("No unprivileged samples (sensitive_attr == 0) found.")

    return (
        (y_true[mask_unpriv], y_pred[mask_unpriv]),
        (y_true[mask_priv], y_pred[mask_priv]),
    )


def _safe_ratio(numer: float, denom: float) -> float:
    """Return numer/denom with the standard fairness-ratio zero-handling.

    - If both numerator and denominator are 0, the groups agree: return 1.0.
    - If only the denominator is 0, the ratio is undefined: return NaN.
    """
    if denom == 0.0:
        return 1.0 if numer == 0.0 else float("nan")
    return float(numer / denom)


# ---------------------------------------------------------------------------
# Positive prediction rate: SPD (diff) / Disparate Impact (ratio)
# ---------------------------------------------------------------------------

def statistical_parity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in positive prediction rates (unprivileged - privileged).

    .. math::
        SPD = P(\\hat{Y}=1 \\mid S=0) - P(\\hat{Y}=1 \\mid S=1)

    A value of 0 indicates perfect fairness. Negative values indicate the
    unprivileged group is disadvantaged.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (_, y_pred_u), (_, y_pred_p) = _split_by_group(y_true, y_pred, sensitive_attr)

    rate_priv = np.mean(y_pred_p == 1)
    rate_unpriv = np.mean(y_pred_u == 1)

    return float(rate_unpriv - rate_priv)


def disparate_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of positive prediction rates (unprivileged / privileged).

    .. math::
        DI = \\frac{P(\\hat{Y}=1 \\mid S=0)}{P(\\hat{Y}=1 \\mid S=1)}

    A value of 1.0 indicates perfect fairness. The 80 % rule threshold is 0.8.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (_, y_pred_u), (_, y_pred_p) = _split_by_group(y_true, y_pred, sensitive_attr)

    rate_priv = float(np.mean(y_pred_p == 1))
    rate_unpriv = float(np.mean(y_pred_u == 1))

    return _safe_ratio(rate_unpriv, rate_priv)


# ---------------------------------------------------------------------------
# TPR: Equal Opportunity Diff / Equal Opportunity Ratio
# ---------------------------------------------------------------------------

def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in true positive rates (unprivileged - privileged).

    .. math::
        EOD = TPR_{\\text{unpriv}} - TPR_{\\text{priv}}

    A value of 0 indicates perfect fairness.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    return float(
        true_positive_rate(y_true_u, y_pred_u)
        - true_positive_rate(y_true_p, y_pred_p)
    )


def equal_opportunity_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of true positive rates (unprivileged / privileged).

    .. math::
        EOR = \\frac{TPR_{\\text{unpriv}}}{TPR_{\\text{priv}}}

    A value of 1.0 indicates perfect fairness.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    return _safe_ratio(
        true_positive_rate(y_true_u, y_pred_u),
        true_positive_rate(y_true_p, y_pred_p),
    )


# ---------------------------------------------------------------------------
# FPR: False Positive Rate Diff / Predictive Equality (FPR ratio)
# ---------------------------------------------------------------------------

def false_positive_rate_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in false positive rates (unprivileged - privileged).

    .. math::
        FPRD = FPR_{\\text{unpriv}} - FPR_{\\text{priv}}

    A value of 0 indicates perfect fairness. Positive values indicate the
    unprivileged group is falsely flagged as positive more often.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    return float(
        false_positive_rate(y_true_u, y_pred_u)
        - false_positive_rate(y_true_p, y_pred_p)
    )


def predictive_equality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of false positive rates (unprivileged / privileged).

    .. math::
        PE = \\frac{FPR_{\\text{unpriv}}}{FPR_{\\text{priv}}}

    Also exported as :func:`false_positive_rate_parity`. A value of 1.0
    indicates perfect fairness.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    return _safe_ratio(
        false_positive_rate(y_true_u, y_pred_u),
        false_positive_rate(y_true_p, y_pred_p),
    )


false_positive_rate_parity = predictive_equality


# ---------------------------------------------------------------------------
# TNR: True Negative Rate Diff / True Negative Rate Parity
# ---------------------------------------------------------------------------

def true_negative_rate_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in true negative rates (unprivileged - privileged).

    .. math::
        TNRD = TNR_{\\text{unpriv}} - TNR_{\\text{priv}}

    A value of 0 indicates perfect fairness.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    return float(
        true_negative_rate(y_true_u, y_pred_u)
        - true_negative_rate(y_true_p, y_pred_p)
    )


def true_negative_rate_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of true negative rates (unprivileged / privileged).

    .. math::
        TNRP = \\frac{TNR_{\\text{unpriv}}}{TNR_{\\text{priv}}}

    A value of 1.0 indicates perfect fairness.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    return _safe_ratio(
        true_negative_rate(y_true_u, y_pred_u),
        true_negative_rate(y_true_p, y_pred_p),
    )


# ---------------------------------------------------------------------------
# FNR: False Negative Rate Diff / False Negative Rate Parity
# ---------------------------------------------------------------------------

def false_negative_rate_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in false negative rates (unprivileged - privileged).

    .. math::
        FNRD = FNR_{\\text{unpriv}} - FNR_{\\text{priv}}

    A value of 0 indicates perfect fairness. Positive values indicate the
    unprivileged group has higher false negative rates.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    return float(
        false_negative_rate(y_true_u, y_pred_u)
        - false_negative_rate(y_true_p, y_pred_p)
    )


def false_negative_rate_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of false negative rates (unprivileged / privileged).

    .. math::
        FNRP = \\frac{FNR_{\\text{unpriv}}}{FNR_{\\text{priv}}}

    A value of 1.0 indicates perfect fairness.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    return _safe_ratio(
        false_negative_rate(y_true_u, y_pred_u),
        false_negative_rate(y_true_p, y_pred_p),
    )


# ---------------------------------------------------------------------------
# Accuracy: Accuracy Difference / Accuracy Parity (ratio)
# ---------------------------------------------------------------------------

def accuracy_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in accuracies (unprivileged - privileged).

    .. math::
        AD = Acc_{\\text{unpriv}} - Acc_{\\text{priv}}

    A value of 0 indicates perfect fairness.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    return float(
        accuracy(y_true_u, y_pred_u)
        - accuracy(y_true_p, y_pred_p)
    )


def accuracy_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of accuracies (unprivileged / privileged).

    .. math::
        AP = \\frac{Acc_{\\text{unpriv}}}{Acc_{\\text{priv}}}

    A value of 1.0 indicates perfect fairness.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    return _safe_ratio(
        accuracy(y_true_u, y_pred_u),
        accuracy(y_true_p, y_pred_p),
    )


# ---------------------------------------------------------------------------
# Combined FPR + TPR: Average Odds Difference / Average Odds Ratio
# ---------------------------------------------------------------------------

def average_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Average of FPR difference and TPR difference across groups.

    .. math::
        AOD = 0.5 \\times [(FPR_{unpriv} - FPR_{priv})
                          + (TPR_{unpriv} - TPR_{priv})]

    A value of 0 indicates perfect fairness.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    fpr_diff = (false_positive_rate(y_true_u, y_pred_u)
                - false_positive_rate(y_true_p, y_pred_p))
    tpr_diff = (true_positive_rate(y_true_u, y_pred_u)
                - true_positive_rate(y_true_p, y_pred_p))
    return float(0.5 * (fpr_diff + tpr_diff))


def average_odds_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Average of FPR ratio and TPR ratio across groups.

    .. math::
        AOR = 0.5 \\times \\left[
            \\frac{FPR_{\\text{unpriv}}}{FPR_{\\text{priv}}}
            + \\frac{TPR_{\\text{unpriv}}}{TPR_{\\text{priv}}}
        \\right]

    A value of 1.0 indicates perfect fairness. Returns NaN if either
    component ratio is undefined (the corresponding privileged-group rate
    is zero while the unprivileged-group rate is positive).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).

    Returns
    -------
    float
    """
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    fpr_ratio = _safe_ratio(
        false_positive_rate(y_true_u, y_pred_u),
        false_positive_rate(y_true_p, y_pred_p),
    )
    tpr_ratio = _safe_ratio(
        true_positive_rate(y_true_u, y_pred_u),
        true_positive_rate(y_true_p, y_pred_p),
    )
    if np.isnan(fpr_ratio) or np.isnan(tpr_ratio):
        return float("nan")
    return float(0.5 * (fpr_ratio + tpr_ratio))
