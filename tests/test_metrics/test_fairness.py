"""Tests for group fairness metrics."""

import numpy as np
import pytest

from skfair.metrics import (
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
        assert np.isnan(disparate_impact([0]*4, y_pred, s_attr))

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


# ---------------------------------------------------------------------------
# New counterpart metrics: each base measure now has both diff and ratio form
# ---------------------------------------------------------------------------

class TestCounterpartPerfect:
    """All metrics should hit their ideal value on perfectly fair data."""

    def test_fpr_difference(self):
        assert false_positive_rate_difference(*_perfect_data()) == 0.0

    def test_false_negative_rate_difference(self):
        assert false_negative_rate_difference(*_perfect_data()) == 0.0

    def test_accuracy_difference(self):
        assert accuracy_difference(*_perfect_data()) == 0.0

    def test_average_odds_ratio(self):
        # Perfect data: TPR=1 for both groups, FPR=0 for both groups.
        # FPR ratio is 0/0 -> 1.0 by _safe_ratio; TPR ratio is 1/1 = 1.0.
        # AOR = 0.5 * (1 + 1) = 1.0
        assert average_odds_ratio(*_perfect_data()) == 1.0

    def test_tnr_parity(self):
        assert true_negative_rate_parity(*_perfect_data()) == 1.0

    def test_fnr_parity(self):
        # Perfect classifier: FNR = 0 for both groups -> 0/0 -> 1.0
        assert false_negative_rate_parity(*_perfect_data()) == 1.0

    def test_fpr_parity_alias(self):
        # Alias should match predictive_equality exactly
        assert false_positive_rate_parity is predictive_equality

    def test_accuracy_parity_perfect(self):
        assert accuracy_parity(*_perfect_data()) == 1.0


class TestCounterpartBiased:
    """On the biased fixture, diff and ratio counterparts agree on direction."""

    def test_fpr_difference_biased(self):
        # Priv: FPR=0, Unpriv: FPR=1 -> diff = +1
        y_true, y_pred, s_attr = _biased_data()
        assert false_positive_rate_difference(y_true, y_pred, s_attr) == 1.0

    def test_fnr_difference_biased(self):
        # Priv: FNR=0, Unpriv: FNR=1 -> diff = +1
        y_true, y_pred, s_attr = _biased_data()
        assert false_negative_rate_difference(y_true, y_pred, s_attr) == 1.0

    def test_accuracy_difference_biased(self):
        # Priv: acc=1, Unpriv: acc=0 -> diff = -1
        y_true, y_pred, s_attr = _biased_data()
        assert accuracy_difference(y_true, y_pred, s_attr) == -1.0

    def test_accuracy_parity_biased(self):
        # acc_unpriv=0 / acc_priv=1 = 0.0
        y_true, y_pred, s_attr = _biased_data()
        assert accuracy_parity(y_true, y_pred, s_attr) == 0.0

    def test_tnr_parity_biased(self):
        # TNR_unpriv=0 / TNR_priv=1 = 0.0
        y_true, y_pred, s_attr = _biased_data()
        assert true_negative_rate_parity(y_true, y_pred, s_attr) == 0.0


class TestCounterpartSymmetry:
    """Diff metrics flip sign when groups are swapped."""

    def test_fpr_difference_sign_flip(self):
        y_true, y_pred, s_attr = _biased_data()
        s_flipped = 1 - np.array(s_attr)
        assert (
            false_positive_rate_difference(y_true, y_pred, s_attr)
            == -false_positive_rate_difference(y_true, y_pred, s_flipped)
        )

    def test_accuracy_difference_sign_flip(self):
        y_true, y_pred, s_attr = _biased_data()
        s_flipped = 1 - np.array(s_attr)
        assert (
            accuracy_difference(y_true, y_pred, s_attr)
            == -accuracy_difference(y_true, y_pred, s_flipped)
        )


class TestCounterpartEdgeCases:
    """Ratio metrics return NaN when the privileged-group rate is zero
    and the unprivileged-group rate is positive, and 1.0 when both are zero."""

    def test_tnr_parity_priv_zero_unpriv_positive(self):
        # Priv all positives in y_true -> TNR_priv = 0; Unpriv has negatives
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        s_attr = [1, 1, 0, 0]
        # TNR_priv = 0 (no priv negatives), TNR_unpriv = 1 (all correct)
        assert np.isnan(true_negative_rate_parity(y_true, y_pred, s_attr))

    def test_fnr_parity_both_zero(self):
        # Both groups have FNR = 0 -> 0/0 -> 1.0
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 1, 0]
        s_attr = [1, 1, 0, 0]
        assert false_negative_rate_parity(y_true, y_pred, s_attr) == 1.0

    def test_average_odds_ratio_nan_propagates(self):
        # Construct data so FPR_priv = 0 but FPR_unpriv > 0 -> NaN
        # Priv: y_true=[1,1], y_pred=[1,1] -> no negatives so FPR undefined (0)
        # Unpriv: y_true=[0,0], y_pred=[1,1] -> FPR = 1
        # ratio is NaN -> AOR is NaN
        import warnings
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 1, 1]
        s_attr = [1, 1, 0, 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert np.isnan(average_odds_ratio(y_true, y_pred, s_attr))

    def test_eor_alias_check(self):
        # equal_opportunity_ratio uses _safe_ratio: TPR=1 priv, TPR=0 unpriv -> 0.0
        y_true, y_pred, s_attr = _biased_data()
        assert equal_opportunity_ratio(y_true, y_pred, s_attr) == 0.0
