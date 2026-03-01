"""Standard binary classification performance metrics."""

import warnings

import numpy as np

__all__ = [
    "accuracy",
    "true_positive_rate",
    "false_positive_rate",
    "true_negative_rate",
    "false_negative_rate",
    "balanced_accuracy",
]


def _confusion_counts(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple:
    """Return (TP, TN, FP, FN) from binary y_true and y_pred.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).

    Returns
    -------
    tuple of (int, int, int, int)
        (TP, TN, FP, FN)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {len(y_true)} and {len(y_pred)}."
        )
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")
    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_pred must contain only 0 and 1.")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).

    Returns
    -------
    float
        ``(TP + TN) / (TP + TN + FP + FN)``
    """
    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    return (tp + tn) / total


def true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute true positive rate (recall / sensitivity).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).

    Returns
    -------
    float
        ``TP / (TP + FN)``, or 0.0 if no actual positives.
    """
    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)
    denom = tp + fn
    if denom == 0:
        warnings.warn("No actual positives; TPR is undefined, returning 0.0.",
                       RuntimeWarning, stacklevel=2)
        return 0.0
    return tp / denom


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute false positive rate (fall-out).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).

    Returns
    -------
    float
        ``FP / (FP + TN)``, or 0.0 if no actual negatives.
    """
    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)
    denom = fp + tn
    if denom == 0:
        warnings.warn("No actual negatives; FPR is undefined, returning 0.0.",
                       RuntimeWarning, stacklevel=2)
        return 0.0
    return fp / denom


def true_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute true negative rate (specificity).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).

    Returns
    -------
    float
        ``TN / (TN + FP)``, or 0.0 if no actual negatives.
    """
    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)
    denom = tn + fp
    if denom == 0:
        warnings.warn("No actual negatives; TNR is undefined, returning 0.0.",
                       RuntimeWarning, stacklevel=2)
        return 0.0
    return tn / denom


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute false negative rate (miss rate).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).

    Returns
    -------
    float
        ``FN / (TP + FN)``, or 0.0 if no actual positives.
    """
    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)
    denom = tp + fn
    if denom == 0:
        warnings.warn("No actual positives; FNR is undefined, returning 0.0.",
                       RuntimeWarning, stacklevel=2)
        return 0.0
    return fn / denom


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).

    Returns
    -------
    float
        ``(TPR + TNR) / 2``
    """
    return (true_positive_rate(y_true, y_pred)
            + true_negative_rate(y_true, y_pred)) / 2
