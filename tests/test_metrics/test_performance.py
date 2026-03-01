"""Tests for performance metrics."""

import warnings

import numpy as np
import pytest

from skfair.metrics import (
    accuracy,
    balanced_accuracy,
    false_negative_rate,
    false_positive_rate,
    true_negative_rate,
    true_positive_rate,
)


class TestAccuracy:
    def test_perfect(self):
        assert accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0

    def test_all_wrong(self):
        assert accuracy([1, 1, 0, 0], [0, 0, 1, 1]) == 0.0

    def test_half(self):
        assert accuracy([1, 1, 0, 0], [1, 0, 0, 1]) == 0.5


class TestTruePositiveRate:
    def test_perfect(self):
        assert true_positive_rate([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0

    def test_zero(self):
        assert true_positive_rate([1, 1, 0, 0], [0, 0, 0, 0]) == 0.0

    def test_no_actual_positives(self):
        with pytest.warns(RuntimeWarning, match="No actual positives"):
            result = true_positive_rate([0, 0], [1, 0])
        assert result == 0.0


class TestFalsePositiveRate:
    def test_perfect(self):
        assert false_positive_rate([1, 1, 0, 0], [1, 1, 0, 0]) == 0.0

    def test_all_false_positives(self):
        assert false_positive_rate([0, 0], [1, 1]) == 1.0

    def test_no_actual_negatives(self):
        with pytest.warns(RuntimeWarning, match="No actual negatives"):
            result = false_positive_rate([1, 1], [1, 0])
        assert result == 0.0


class TestTrueNegativeRate:
    def test_perfect(self):
        assert true_negative_rate([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0

    def test_zero(self):
        assert true_negative_rate([0, 0], [1, 1]) == 0.0


class TestFalseNegativeRate:
    def test_perfect(self):
        assert false_negative_rate([1, 1, 0, 0], [1, 1, 0, 0]) == 0.0

    def test_all_missed(self):
        assert false_negative_rate([1, 1], [0, 0]) == 1.0


class TestBalancedAccuracy:
    def test_perfect(self):
        assert balanced_accuracy([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0

    def test_random(self):
        # Predicts all 1 -> TPR=1, TNR=0 -> BA=0.5
        assert balanced_accuracy([1, 1, 0, 0], [1, 1, 1, 1]) == 0.5


class TestInputValidation:
    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            accuracy([1, 0], [1])

    def test_invalid_y_true(self):
        with pytest.raises(ValueError, match="y_true must contain only 0 and 1"):
            accuracy([2, 0], [1, 0])

    def test_invalid_y_pred(self):
        with pytest.raises(ValueError, match="y_pred must contain only 0 and 1"):
            accuracy([1, 0], [1, 3])
