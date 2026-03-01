"""Tests for group fairness metrics."""

import numpy as np
import pytest

from skfair.metrics import (
    average_odds_difference,
    disparate_impact,
    equal_opportunity_difference,
    statistical_parity_difference,
    true_negative_rate_difference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect_data():
    """Perfect classifier, balanced groups."""
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    s_attr = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    return y_true, y_pred, s_attr


def _biased_data():
    """Classifier that favours the privileged group.

    Priv (s=1): y_true=[1,1,0,0], y_pred=[1,1,0,0]  -> all correct
    Unpriv (s=0): y_true=[1,1,0,0], y_pred=[0,0,1,1] -> all wrong
    """
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    s_attr = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    return y_true, y_pred, s_attr


# ---------------------------------------------------------------------------
# Perfect classifier
# ---------------------------------------------------------------------------

class TestPerfectClassifier:
    def test_spd(self):
        assert statistical_parity_difference(*_perfect_data()) == 0.0

    def test_di(self):
        assert disparate_impact(*_perfect_data()) == 1.0

    def test_eod(self):
        assert equal_opportunity_difference(*_perfect_data()) == 0.0

    def test_aod(self):
        assert average_odds_difference(*_perfect_data()) == 0.0

    def test_tnrd(self):
        assert true_negative_rate_difference(*_perfect_data()) == 0.0


# ---------------------------------------------------------------------------
# Biased classifier (favours privileged)
# ---------------------------------------------------------------------------

class TestBiasedClassifier:
    def test_spd_negative(self):
        # Priv selection rate = 0.5, unpriv = 0.5 — actually same rate
        # Let's use a clearer example
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 0, 0]
        s_attr = [1, 1, 0, 0]
        assert statistical_parity_difference(y_true, y_pred, s_attr) == -1.0

    def test_di_below_one(self):
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 0, 0]
        s_attr = [1, 1, 0, 0]
        assert disparate_impact(y_true, y_pred, s_attr) == 0.0

    def test_eod_negative(self):
        # Priv: TPR=1, Unpriv: TPR=0
        y_true, y_pred, s_attr = _biased_data()
        assert equal_opportunity_difference(y_true, y_pred, s_attr) == -1.0

    def test_aod_negative(self):
        y_true, y_pred, s_attr = _biased_data()
        result = average_odds_difference(y_true, y_pred, s_attr)
        # FPR_unpriv=1, FPR_priv=0 -> diff=+1
        # TPR_unpriv=0, TPR_priv=1 -> diff=-1
        # AOD = 0.5*(1 + (-1)) = 0
        assert result == 0.0

    def test_tnrd(self):
        y_true, y_pred, s_attr = _biased_data()
        # TNR_priv=1, TNR_unpriv=0
        assert true_negative_rate_difference(y_true, y_pred, s_attr) == -1.0


# ---------------------------------------------------------------------------
# Symmetry: swapping priv/unpriv flips sign
# ---------------------------------------------------------------------------

class TestSymmetry:
    def test_spd_sign_flip(self):
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 0, 0]
        s1 = [1, 1, 0, 0]
        s2 = [0, 0, 1, 1]  # swapped
        assert (
            statistical_parity_difference(y_true, y_pred, s1)
            == -statistical_parity_difference(y_true, y_pred, s2)
        )

    def test_eod_sign_flip(self):
        y_true, y_pred, s_attr = _biased_data()
        s_flipped = 1 - np.array(s_attr)
        assert (
            equal_opportunity_difference(y_true, y_pred, s_attr)
            == -equal_opportunity_difference(y_true, y_pred, s_flipped)
        )

    def test_tnrd_sign_flip(self):
        y_true, y_pred, s_attr = _biased_data()
        s_flipped = 1 - np.array(s_attr)
        assert (
            true_negative_rate_difference(y_true, y_pred, s_attr)
            == -true_negative_rate_difference(y_true, y_pred, s_flipped)
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_privileged_group(self):
        with pytest.raises(ValueError, match="No privileged"):
            statistical_parity_difference([1, 0], [1, 0], [0, 0])

    def test_empty_unprivileged_group(self):
        with pytest.raises(ValueError, match="No unprivileged"):
            statistical_parity_difference([1, 0], [1, 0], [1, 1])

    def test_di_priv_rate_zero_unpriv_positive(self):
        # Priv all predict 0, unpriv all predict 1
        y_pred = [0, 0, 1, 1]
        s_attr = [1, 1, 0, 0]
        assert disparate_impact([0]*4, y_pred, s_attr) == float("inf")

    def test_di_both_rates_zero(self):
        y_pred = [0, 0, 0, 0]
        s_attr = [1, 1, 0, 0]
        assert disparate_impact([0]*4, y_pred, s_attr) == 1.0

    def test_single_class_group(self):
        # Unpriv group has only positives in y_true
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 1, 1]
        s_attr = [1, 1, 0, 0]
        # Should not raise — TNR for unpriv will warn (no negatives) but return 0
        result = true_negative_rate_difference(y_true, y_pred, s_attr)
        assert isinstance(result, float)
