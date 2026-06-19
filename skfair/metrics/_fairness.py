"""Group fairness metrics for binary classification.

Every metric compares an *unprivileged* group (``sensitive_attr == 0``) to a
*privileged* group (``sensitive_attr == 1``) on some base measure, and is
exposed in two counterpart forms:

* a **difference** form (suffix ``_difference``, ideal = 0), and
* a **ratio** form (suffix ``_ratio``, ideal = 1).

Naming policy
-------------
The canonical/primary name of every metric is the full descriptive form with a
``_difference`` or ``_ratio`` suffix (never ``_parity``) — e.g.
``false_positive_rate_difference``, ``positive_predictive_value_ratio``,
``average_odds_difference``. Each metric also carries short aliases: the
mechanical short form (``fpr_diff``, ``ppv_ratio``) and, where one exists, the
literature abbreviation or criterion name (``eod``, ``eor``, ``spd``, ``di``,
``aod``, ``aor``, ``predictive_parity_ratio``). Aliases are plain assignments
and are exported in ``__all__``.

Metrics are grouped into four fairness *families*:

* **independence** — depends on the prediction Ŷ only (SPD, disparate impact).
* **separation** — conditions on the true label Y (TPR/FPR/TNR/FNR, average odds).
* **sufficiency** — conditions on the prediction Ŷ (PPV/NPV/FDR/FOR); the same
  confusion-matrix cells read along the prediction axis instead of the truth
  axis.
* **accuracy** — group accuracy difference/ratio.

A :data:`METRICS` registry maps each primary name to its function plus metadata
(family and ideal value), and :func:`benchmark` evaluates a selection of metrics
in one call. The full grid of ~22 metrics is available for completeness, but
``benchmark`` defaults to a sensible one-per-family subset (see
:data:`DEFAULT_BENCHMARK_METRICS`).

Fairness functions import the corresponding base measures from
``_performance.py`` so that formulas read like their mathematical definitions.
"""

from collections import namedtuple

import numpy as np

from ._performance import (
    accuracy,
    false_discovery_rate,
    false_negative_rate,
    false_omission_rate,
    false_positive_rate,
    negative_predictive_value,
    positive_predictive_value,
    true_negative_rate,
    true_positive_rate,
)

__all__ = [
    # ----- independence (depends on Ŷ only) -----
    "statistical_parity_difference",
    "disparate_impact",
    "spd",
    "di",
    # ----- separation: TPR -----
    "true_positive_rate_difference",
    "true_positive_rate_ratio",
    "eod",
    "tpr_diff",
    "eor",
    "tpr_ratio",
    "equal_opportunity_difference",  # legacy alias
    "equal_opportunity_ratio",       # legacy alias
    # ----- separation: FPR -----
    "false_positive_rate_difference",
    "false_positive_rate_ratio",
    "fpr_diff",
    "fpr_ratio",
    # ----- separation: TNR -----
    "true_negative_rate_difference",
    "true_negative_rate_ratio",
    "tnr_diff",
    "tnr_ratio",
    # ----- separation: FNR -----
    "false_negative_rate_difference",
    "false_negative_rate_ratio",
    "fnr_diff",
    "fnr_ratio",
    # ----- separation: combined odds -----
    "average_odds_difference",
    "average_odds_ratio",
    "aod",
    "aor",
    # ----- sufficiency: PPV -----
    "positive_predictive_value_difference",
    "positive_predictive_value_ratio",
    "ppv_diff",
    "ppv_ratio",
    "predictive_parity_difference",
    "predictive_parity_ratio",
    # ----- sufficiency: NPV -----
    "negative_predictive_value_difference",
    "negative_predictive_value_ratio",
    "npv_diff",
    "npv_ratio",
    # ----- sufficiency: FDR -----
    "false_discovery_rate_difference",
    "false_discovery_rate_ratio",
    "fdr_diff",
    "fdr_ratio",
    # ----- sufficiency: FOR -----
    "false_omission_rate_difference",
    "false_omission_rate_ratio",
    "for_diff",
    "for_ratio",
    # ----- accuracy -----
    "accuracy_difference",
    "accuracy_ratio",
    "acc_diff",
    "acc_ratio",
    # ----- registry / benchmark -----
    "METRICS",
    "DEFAULT_BENCHMARK_METRICS",
    "benchmark",
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


def _group_difference(base_fn, y_true, y_pred, sensitive_attr) -> float:
    """Unprivileged - privileged difference of a base measure (ideal = 0)."""
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    return float(base_fn(y_true_u, y_pred_u) - base_fn(y_true_p, y_pred_p))


def _group_ratio(base_fn, y_true, y_pred, sensitive_attr) -> float:
    """Unprivileged / privileged ratio of a base measure (ideal = 1)."""
    (y_true_u, y_pred_u), (y_true_p, y_pred_p) = _split_by_group(
        y_true, y_pred, sensitive_attr
    )
    return _safe_ratio(base_fn(y_true_u, y_pred_u), base_fn(y_true_p, y_pred_p))


# ===========================================================================
# Independence family — depends on the prediction Ŷ only
# ===========================================================================

def statistical_parity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in positive prediction rates (unprivileged - privileged).

    Aliased as ``spd``.

    .. math::
        SPD = P(\\hat{Y}=1 \\mid S=0) - P(\\hat{Y}=1 \\mid S=1)

    A value of 0 indicates perfect fairness. Negative values indicate the
    unprivileged group is selected for the positive outcome less often.

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
    return float(np.mean(y_pred_u == 1) - np.mean(y_pred_p == 1))


def disparate_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of positive prediction rates (unprivileged / privileged).

    Aliased as ``di``.

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
    return _safe_ratio(float(np.mean(y_pred_u == 1)), float(np.mean(y_pred_p == 1)))


# ===========================================================================
# Separation family — TPR (true positive rate / recall)
# ===========================================================================

def true_positive_rate_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in true positive rates (unprivileged - privileged).

    The *equal opportunity* criterion. Aliased as ``eod``, ``tpr_diff`` and
    (legacy) ``equal_opportunity_difference``.

    .. math::
        TPRD = TPR_{\\text{unpriv}} - TPR_{\\text{priv}}

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
    return _group_difference(true_positive_rate, y_true, y_pred, sensitive_attr)


def true_positive_rate_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of true positive rates (unprivileged / privileged).

    Aliased as ``eor``, ``tpr_ratio`` and (legacy) ``equal_opportunity_ratio``.

    .. math::
        TPRR = \\frac{TPR_{\\text{unpriv}}}{TPR_{\\text{priv}}}

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
    return _group_ratio(true_positive_rate, y_true, y_pred, sensitive_attr)


# ===========================================================================
# Separation family — FPR (false positive rate)
# ===========================================================================

def false_positive_rate_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in false positive rates (unprivileged - privileged).

    Aliased as ``fpr_diff``.

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
    return _group_difference(false_positive_rate, y_true, y_pred, sensitive_attr)


def false_positive_rate_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of false positive rates (unprivileged / privileged).

    Aliased as ``fpr_ratio``.

    .. math::
        FPRR = \\frac{FPR_{\\text{unpriv}}}{FPR_{\\text{priv}}}

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
    return _group_ratio(false_positive_rate, y_true, y_pred, sensitive_attr)


# ===========================================================================
# Separation family — TNR (true negative rate / specificity)
# ===========================================================================

def true_negative_rate_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in true negative rates (unprivileged - privileged).

    Aliased as ``tnr_diff``. Implemented as the sign-flip of
    :func:`false_positive_rate_difference`.

    .. math::
        TNRD = TNR_{\\text{unpriv}} - TNR_{\\text{priv}} = -FPRD

    **This difference is redundant.** Since ``TNR = 1 - FPR``, the TNR gap is
    exactly the negated FPR gap, so it carries no information beyond
    :func:`false_positive_rate_difference`. It is kept deliberately for two
    reasons: (a) completeness/symmetry of the difference+ratio grid, and
    (b) its *ratio* counterpart is **not** redundant —
    :func:`true_negative_rate_ratio` is not any function of
    :func:`false_positive_rate_ratio`, because a ratio of ``1 - FPR`` values is
    not a function of the ratio of the ``FPR`` values. The differences are
    redundant; the ratios are not.

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
    return -false_positive_rate_difference(y_true, y_pred, sensitive_attr)


def true_negative_rate_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of true negative rates (unprivileged / privileged).

    Aliased as ``tnr_ratio``. Unlike its difference counterpart, this ratio is
    **not** a sign-flip or any function of :func:`false_positive_rate_ratio` —
    it carries genuinely new information.

    .. math::
        TNRR = \\frac{TNR_{\\text{unpriv}}}{TNR_{\\text{priv}}}

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
    return _group_ratio(true_negative_rate, y_true, y_pred, sensitive_attr)


# ===========================================================================
# Separation family — FNR (false negative rate / miss rate)
# ===========================================================================

def false_negative_rate_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in false negative rates (unprivileged - privileged).

    Aliased as ``fnr_diff``. Implemented as the sign-flip of
    :func:`true_positive_rate_difference` (``eod``).

    .. math::
        FNRD = FNR_{\\text{unpriv}} - FNR_{\\text{priv}} = -TPRD

    **This difference is redundant.** Since ``FNR = 1 - TPR``, the FNR gap is
    exactly the negated TPR gap and adds nothing beyond
    :func:`true_positive_rate_difference`. It is kept deliberately for
    (a) completeness/symmetry of the difference+ratio grid, and (b) because its
    *ratio* counterpart :func:`false_negative_rate_ratio` is **not** redundant —
    it is not any function of :func:`true_positive_rate_ratio`. The differences
    are redundant; the ratios are not.

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
    return -true_positive_rate_difference(y_true, y_pred, sensitive_attr)


def false_negative_rate_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of false negative rates (unprivileged / privileged).

    Aliased as ``fnr_ratio``. Unlike its difference counterpart, this ratio is
    **not** a sign-flip or any function of :func:`true_positive_rate_ratio` —
    it carries genuinely new information.

    .. math::
        FNRR = \\frac{FNR_{\\text{unpriv}}}{FNR_{\\text{priv}}}

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
    return _group_ratio(false_negative_rate, y_true, y_pred, sensitive_attr)


# ===========================================================================
# Separation family — combined odds (FPR + TPR)
# ===========================================================================

def average_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Average of FPR difference and TPR difference across groups.

    Aliased as ``aod``.

    .. math::
        AOD = 0.5 \\times [(FPR_{unpriv} - FPR_{priv})
                          + (TPR_{unpriv} - TPR_{priv})]

    A value of 0 indicates perfect fairness. The TPR and FPR gaps shown side by
    side already represent equalized odds (AIF360 convention); no separate
    max-scalar ``equalized_odds_difference`` is provided.

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
    fpr_d = false_positive_rate_difference(y_true, y_pred, sensitive_attr)
    tpr_d = true_positive_rate_difference(y_true, y_pred, sensitive_attr)
    return float(0.5 * (fpr_d + tpr_d))


def average_odds_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Average of FPR ratio and TPR ratio across groups.

    Aliased as ``aor``.

    .. math::
        AOR = 0.5 \\times \\left[
            \\frac{FPR_{\\text{unpriv}}}{FPR_{\\text{priv}}}
            + \\frac{TPR_{\\text{unpriv}}}{TPR_{\\text{priv}}}
        \\right]

    A value of 1.0 indicates perfect fairness. Returns NaN if either component
    ratio is undefined (the corresponding privileged-group rate is zero while
    the unprivileged-group rate is positive).

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
    fpr_r = false_positive_rate_ratio(y_true, y_pred, sensitive_attr)
    tpr_r = true_positive_rate_ratio(y_true, y_pred, sensitive_attr)
    if np.isnan(fpr_r) or np.isnan(tpr_r):
        return float("nan")
    return float(0.5 * (fpr_r + tpr_r))


# ===========================================================================
# Sufficiency family — condition on the prediction Ŷ, not the true label Y.
# These read the same confusion-matrix cells along the prediction axis
# (Ŷ=1 -> PPV/FDR, Ŷ=0 -> NPV/FOR) instead of the truth axis used by the
# separation family above.
# ===========================================================================

def positive_predictive_value_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in positive predictive value (unprivileged - privileged).

    The *predictive parity* criterion. Aliased as ``ppv_diff`` and
    ``predictive_parity_difference``. Conditions on the prediction Ŷ=1 (the
    precision axis), not on the true label Y.

    .. math::
        PPVD = PPV_{\\text{unpriv}} - PPV_{\\text{priv}}

    A value of 0 indicates perfect fairness. NaN if a group has no predicted
    positives.

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
    return _group_difference(positive_predictive_value, y_true, y_pred, sensitive_attr)


def positive_predictive_value_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of positive predictive value (unprivileged / privileged).

    Aliased as ``ppv_ratio`` and ``predictive_parity_ratio``. Conditions on the
    prediction Ŷ=1, not on the true label Y.

    .. math::
        PPVR = \\frac{PPV_{\\text{unpriv}}}{PPV_{\\text{priv}}}

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
    return _group_ratio(positive_predictive_value, y_true, y_pred, sensitive_attr)


def negative_predictive_value_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in negative predictive value (unprivileged - privileged).

    Aliased as ``npv_diff``. Conditions on the prediction Ŷ=0, not on the true
    label Y.

    .. math::
        NPVD = NPV_{\\text{unpriv}} - NPV_{\\text{priv}}

    A value of 0 indicates perfect fairness. NaN if a group has no predicted
    negatives.

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
    return _group_difference(negative_predictive_value, y_true, y_pred, sensitive_attr)


def negative_predictive_value_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of negative predictive value (unprivileged / privileged).

    Aliased as ``npv_ratio``. Conditions on the prediction Ŷ=0, not on the true
    label Y.

    .. math::
        NPVR = \\frac{NPV_{\\text{unpriv}}}{NPV_{\\text{priv}}}

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
    return _group_ratio(negative_predictive_value, y_true, y_pred, sensitive_attr)


def false_discovery_rate_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in false discovery rate (unprivileged - privileged).

    Aliased as ``fdr_diff``. Implemented as the sign-flip of
    :func:`positive_predictive_value_difference`. Conditions on the prediction
    Ŷ=1, not on the true label Y.

    .. math::
        FDRD = FDR_{\\text{unpriv}} - FDR_{\\text{priv}} = -PPVD

    **This difference is redundant.** Since ``FDR = 1 - PPV``, the FDR gap is
    exactly the negated PPV gap and adds nothing beyond
    :func:`positive_predictive_value_difference`. It is kept deliberately for
    (a) completeness/symmetry of the difference+ratio grid, and (b) because its
    *ratio* counterpart :func:`false_discovery_rate_ratio` is **not** redundant —
    it is not any function of :func:`positive_predictive_value_ratio`. The
    differences are redundant; the ratios are not.

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
    return -positive_predictive_value_difference(y_true, y_pred, sensitive_attr)


def false_discovery_rate_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of false discovery rate (unprivileged / privileged).

    Aliased as ``fdr_ratio``. Unlike its difference counterpart, this ratio is
    **not** a sign-flip or any function of
    :func:`positive_predictive_value_ratio` — it carries genuinely new
    information. Conditions on the prediction Ŷ=1, not on the true label Y.

    .. math::
        FDRR = \\frac{FDR_{\\text{unpriv}}}{FDR_{\\text{priv}}}

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
    return _group_ratio(false_discovery_rate, y_true, y_pred, sensitive_attr)


def false_omission_rate_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in false omission rate (unprivileged - privileged).

    Aliased as ``for_diff``. Implemented as the sign-flip of
    :func:`negative_predictive_value_difference`. Conditions on the prediction
    Ŷ=0, not on the true label Y.

    .. math::
        FORD = FOR_{\\text{unpriv}} - FOR_{\\text{priv}} = -NPVD

    **This difference is redundant.** Since ``FOR = 1 - NPV``, the FOR gap is
    exactly the negated NPV gap and adds nothing beyond
    :func:`negative_predictive_value_difference`. It is kept deliberately for
    (a) completeness/symmetry of the difference+ratio grid, and (b) because its
    *ratio* counterpart :func:`false_omission_rate_ratio` is **not** redundant —
    it is not any function of :func:`negative_predictive_value_ratio`. The
    differences are redundant; the ratios are not.

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
    return -negative_predictive_value_difference(y_true, y_pred, sensitive_attr)


def false_omission_rate_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of false omission rate (unprivileged / privileged).

    Aliased as ``for_ratio``. Unlike its difference counterpart, this ratio is
    **not** a sign-flip or any function of
    :func:`negative_predictive_value_ratio` — it carries genuinely new
    information. Conditions on the prediction Ŷ=0, not on the true label Y.

    .. math::
        FORR = \\frac{FOR_{\\text{unpriv}}}{FOR_{\\text{priv}}}

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
    return _group_ratio(false_omission_rate, y_true, y_pred, sensitive_attr)


# ===========================================================================
# Accuracy family
# ===========================================================================

def accuracy_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Difference in accuracies (unprivileged - privileged).

    Aliased as ``acc_diff``.

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
    return _group_difference(accuracy, y_true, y_pred, sensitive_attr)


def accuracy_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Ratio of accuracies (unprivileged / privileged).

    Aliased as ``acc_ratio``.

    .. math::
        AR = \\frac{Acc_{\\text{unpriv}}}{Acc_{\\text{priv}}}

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
    return _group_ratio(accuracy, y_true, y_pred, sensitive_attr)


# ===========================================================================
# Aliases — mechanical short forms and literature abbreviations / criterion
# names. Never a ``_parity`` suffix.
# ===========================================================================

# independence
spd = statistical_parity_difference
di = disparate_impact

# separation: TPR
eod = true_positive_rate_difference
tpr_diff = true_positive_rate_difference
equal_opportunity_difference = true_positive_rate_difference  
eor = true_positive_rate_ratio
tpr_ratio = true_positive_rate_ratio
equal_opportunity_ratio = true_positive_rate_ratio  
# separation: FPR
fpr_diff = false_positive_rate_difference
fpr_ratio = false_positive_rate_ratio

# separation: TNR
tnr_diff = true_negative_rate_difference
tnr_ratio = true_negative_rate_ratio

# separation: FNR
fnr_diff = false_negative_rate_difference
fnr_ratio = false_negative_rate_ratio

# separation: combined odds
aod = average_odds_difference
aor = average_odds_ratio

# sufficiency: PPV
ppv_diff = positive_predictive_value_difference
ppv_ratio = positive_predictive_value_ratio

# sufficiency: NPV
npv_diff = negative_predictive_value_difference
npv_ratio = negative_predictive_value_ratio

# sufficiency: FDR
fdr_diff = false_discovery_rate_difference
fdr_ratio = false_discovery_rate_ratio

# sufficiency: FOR
for_diff = false_omission_rate_difference
for_ratio = false_omission_rate_ratio

# accuracy
acc_diff = accuracy_difference
acc_ratio = accuracy_ratio


# ===========================================================================
# Registry + benchmark wrapper
# ===========================================================================

MetricSpec = namedtuple("MetricSpec", ["func", "family", "ideal"])

#: Registry of every primary fairness metric -> (function, family, ideal value).
#: ``family`` is one of "independence", "separation", "sufficiency",
#: "accuracy"; ``ideal`` is 0 for difference metrics and 1 for ratio metrics.
METRICS = {
    # independence
    "statistical_parity_difference": MetricSpec(statistical_parity_difference, "independence", 0),
    "disparate_impact": MetricSpec(disparate_impact, "independence", 1),
    # separation
    "true_positive_rate_difference": MetricSpec(true_positive_rate_difference, "separation", 0),
    "true_positive_rate_ratio": MetricSpec(true_positive_rate_ratio, "separation", 1),
    "false_positive_rate_difference": MetricSpec(false_positive_rate_difference, "separation", 0),
    "false_positive_rate_ratio": MetricSpec(false_positive_rate_ratio, "separation", 1),
    "true_negative_rate_difference": MetricSpec(true_negative_rate_difference, "separation", 0),
    "true_negative_rate_ratio": MetricSpec(true_negative_rate_ratio, "separation", 1),
    "false_negative_rate_difference": MetricSpec(false_negative_rate_difference, "separation", 0),
    "false_negative_rate_ratio": MetricSpec(false_negative_rate_ratio, "separation", 1),
    "average_odds_difference": MetricSpec(average_odds_difference, "separation", 0),
    "average_odds_ratio": MetricSpec(average_odds_ratio, "separation", 1),
    # sufficiency
    "positive_predictive_value_difference": MetricSpec(positive_predictive_value_difference, "sufficiency", 0),
    "positive_predictive_value_ratio": MetricSpec(positive_predictive_value_ratio, "sufficiency", 1),
    "negative_predictive_value_difference": MetricSpec(negative_predictive_value_difference, "sufficiency", 0),
    "negative_predictive_value_ratio": MetricSpec(negative_predictive_value_ratio, "sufficiency", 1),
    "false_discovery_rate_difference": MetricSpec(false_discovery_rate_difference, "sufficiency", 0),
    "false_discovery_rate_ratio": MetricSpec(false_discovery_rate_ratio, "sufficiency", 1),
    "false_omission_rate_difference": MetricSpec(false_omission_rate_difference, "sufficiency", 0),
    "false_omission_rate_ratio": MetricSpec(false_omission_rate_ratio, "sufficiency", 1),
    # accuracy
    "accuracy_difference": MetricSpec(accuracy_difference, "accuracy", 0),
    "accuracy_ratio": MetricSpec(accuracy_ratio, "accuracy", 1),
}

#: Default subset evaluated by :func:`benchmark` when no ``metrics`` are given.
#: One representative per family (independence keeps both its members), chosen
#: as a sensible default. The full ~22-metric grid is available for
#: completeness via ``metrics="all"`` or by naming metrics explicitly — the
#: design deliberately ships every metric while defaulting to a subset wherever
#: a default is needed.
DEFAULT_BENCHMARK_METRICS = [
    "statistical_parity_difference",  # independence
    "disparate_impact",               # independence
    "true_positive_rate_difference",  # separation
    "false_positive_rate_ratio",      # separation
    "positive_predictive_value_ratio",  # sufficiency
    "accuracy_ratio",                 # accuracy
]


def _resolve_metric(name):
    """Resolve a metric name (primary or alias) to its callable."""
    func = globals().get(name)
    if not callable(func):
        raise KeyError(
            f"Unknown fairness metric '{name}'. "
            f"Use a primary name from METRICS or a documented alias."
        )
    return func


def benchmark(y_true, y_pred, sensitive_attr, metrics=None):
    """Evaluate a selection of group-fairness metrics in one call.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).
    metrics : None, "all", or iterable of str, default=None
        Which metrics to compute.

        * ``None`` — the sensible default subset
          (:data:`DEFAULT_BENCHMARK_METRICS`), one representative per family.
        * ``"all"`` — the complete grid of every metric in :data:`METRICS`.
        * an iterable of names — those metrics, given by primary name or any
          documented alias.

    Returns
    -------
    dict of str -> float
        Mapping of the requested metric names to their values.
    """
    if metrics is None:
        names = list(DEFAULT_BENCHMARK_METRICS)
    elif metrics == "all":
        names = list(METRICS)
    else:
        names = list(metrics)

    return {
        name: _resolve_metric(name)(y_true, y_pred, sensitive_attr)
        for name in names
    }
