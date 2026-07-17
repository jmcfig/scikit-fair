"""Unit tests for the audit module: BiasAuditor and FairnessAuditor."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from skfair.audit import BiasAuditor, FairnessAuditor
from skfair.metrics import METRICS


@pytest.fixture
def audit_data():
    """Small dataset with a clear group disparity for auditing."""
    rng = np.random.RandomState(0)
    n = 200
    group = rng.randint(0, 2, size=n)  # 0 = unprivileged, 1 = privileged
    # Privileged group gets more positive outcomes.
    y = np.where(group == 1, rng.binomial(1, 0.7, n), rng.binomial(1, 0.3, n))
    X = pd.DataFrame({
        "race": group,
        "age": rng.randint(18, 70, n),
        "score": rng.normal(0, 1, n),
    })
    return X, y


@pytest.fixture
def pred_data():
    """Ground-truth, predictions and group indicator for fairness auditing."""
    rng = np.random.RandomState(1)
    n = 200
    group = rng.randint(0, 2, size=n)
    y_true = rng.binomial(1, 0.5, n)
    y_pred = rng.binomial(1, 0.5, n)
    return y_true, y_pred, group


# ---------------------------------------------------------------------------
# BiasAuditor
# ---------------------------------------------------------------------------

class TestBiasAuditor:
    def test_init_stores_attributes(self, audit_data):
        X, y = audit_data
        auditor = BiasAuditor(X, y, sens_attr="race")
        assert auditor.sens_attr == "race"
        assert auditor.priv_group == 1
        assert auditor.pos_label == 1
        assert len(auditor.y) == len(X)

    def test_init_rejects_non_dataframe(self, audit_data):
        _, y = audit_data
        with pytest.raises(TypeError):
            BiasAuditor(np.zeros((10, 2)), y[:10], sens_attr="race")

    def test_init_rejects_missing_sens_attr(self, audit_data):
        X, y = audit_data
        with pytest.raises(ValueError):
            BiasAuditor(X, y, sens_attr="not_a_column")

    def test_group_proportions_sums_to_one(self, audit_data):
        X, y = audit_data
        props = BiasAuditor(X, y, sens_attr="race").group_proportions()
        assert set(props.columns) == {"count", "proportion"}
        assert props["proportion"].sum() == pytest.approx(1.0)
        assert props["count"].sum() == len(X)

    def test_group_proportions_covers_all_groups(self, audit_data):
        X, y = audit_data
        props = BiasAuditor(X, y, sens_attr="race").group_proportions()
        assert set(props.index) == {0, 1}

    def test_target_rate_by_group_shape_and_range(self, audit_data):
        X, y = audit_data
        rates = BiasAuditor(X, y, sens_attr="race").target_rate_by_group()
        assert set(rates.columns) == {"count", "positive_rate"}
        assert ((rates["positive_rate"] >= 0) & (rates["positive_rate"] <= 1)).all()

    def test_target_rate_detects_disparity(self, audit_data):
        X, y = audit_data
        rates = BiasAuditor(X, y, sens_attr="race").target_rate_by_group()
        # Privileged group (1) was constructed with a higher positive rate.
        assert rates.loc[1, "positive_rate"] > rates.loc[0, "positive_rate"]

    def test_target_rate_respects_pos_label(self, audit_data):
        X, y = audit_data
        a1 = BiasAuditor(X, y, sens_attr="race", pos_label=1).target_rate_by_group()
        a0 = BiasAuditor(X, y, sens_attr="race", pos_label=0).target_rate_by_group()
        # Rates for the two opposite labels are complementary.
        np.testing.assert_allclose(
            a1["positive_rate"].values, 1 - a0["positive_rate"].values
        )

    def test_plot_group_proportions_returns_fig_ax(self, audit_data):
        X, y = audit_data
        fig, ax = BiasAuditor(X, y, sens_attr="race").plot_group_proportions()
        assert fig is not None and ax is not None
        plt.close("all")

    def test_plot_target_rates_returns_fig_ax(self, audit_data):
        X, y = audit_data
        fig, ax = BiasAuditor(X, y, sens_attr="race").plot_target_rates()
        assert fig is not None and ax is not None
        plt.close("all")

    def test_plot_feature_distribution_bad_feature_raises(self, audit_data):
        X, y = audit_data
        auditor = BiasAuditor(X, y, sens_attr="race")
        with pytest.raises(ValueError):
            auditor.plot_feature_distribution("nonexistent")

    def test_plot_summary_returns_list(self, audit_data):
        X, y = audit_data
        figs = BiasAuditor(X, y, sens_attr="race").plot_summary()
        assert isinstance(figs, list)
        # group proportions + target rates + one per numeric non-sens feature (age, score)
        assert len(figs) == 4
        plt.close("all")


# ---------------------------------------------------------------------------
# FairnessAuditor
# ---------------------------------------------------------------------------

class TestFairnessAuditor:
    def test_init_builds_binary_mask(self, pred_data):
        yt, yp, s = pred_data
        auditor = FairnessAuditor(yt, yp, s, priv_group=1)
        assert set(np.unique(auditor._sens_binary)).issubset({0, 1})
        assert len(auditor._sens_binary) == len(yt)

    def test_performance_by_group_structure(self, pred_data):
        yt, yp, s = pred_data
        df = FairnessAuditor(yt, yp, s).performance_by_group()
        assert list(df.columns) == ["unprivileged", "privileged"]
        assert set(df.index) == {"Accuracy", "TPR", "FPR", "TNR", "FNR"}

    def test_performance_values_in_unit_range(self, pred_data):
        yt, yp, s = pred_data
        df = FairnessAuditor(yt, yp, s).performance_by_group()
        assert ((df >= 0) & (df <= 1)).all().all()

    def test_fairness_metrics_covers_registry(self, pred_data):
        yt, yp, s = pred_data
        df = FairnessAuditor(yt, yp, s).fairness_metrics()
        expected = {spec.display for spec in METRICS.values()}
        assert set(df.index) == expected
        assert list(df.columns) == ["value"]

    def test_fairness_metrics_values_finite(self, pred_data):
        yt, yp, s = pred_data
        df = FairnessAuditor(yt, yp, s).fairness_metrics()
        assert np.isfinite(df["value"].values).all()

    def test_priv_group_choice_flips_groups(self, pred_data):
        yt, yp, s = pred_data
        a = FairnessAuditor(yt, yp, s, priv_group=1).performance_by_group()
        b = FairnessAuditor(yt, yp, s, priv_group=0).performance_by_group()
        # Swapping the privileged label swaps the two columns.
        np.testing.assert_allclose(
            a["privileged"].values, b["unprivileged"].values
        )

    def test_plot_performance_by_group_returns_fig_ax(self, pred_data):
        yt, yp, s = pred_data
        fig, ax = FairnessAuditor(yt, yp, s).plot_performance_by_group()
        assert fig is not None and ax is not None
        plt.close("all")

    def test_plot_fairness_metrics_returns_fig_ax(self, pred_data):
        yt, yp, s = pred_data
        fig, ax = FairnessAuditor(yt, yp, s).plot_fairness_metrics()
        assert fig is not None and ax is not None
        plt.close("all")

    @pytest.mark.parametrize("mode", ["ratio", "difference", "all"])
    def test_plot_fairness_radar_modes(self, pred_data, mode):
        yt, yp, s = pred_data
        fig, ax = FairnessAuditor(yt, yp, s).plot_fairness_radar(mode=mode)
        assert fig is not None and ax is not None
        plt.close("all")

    def test_plot_summary_returns_three_plots(self, pred_data):
        yt, yp, s = pred_data
        figs = FairnessAuditor(yt, yp, s).plot_summary()
        assert isinstance(figs, list) and len(figs) == 3
        plt.close("all")

    def test_accepts_list_inputs(self, pred_data):
        yt, yp, s = pred_data
        auditor = FairnessAuditor(list(yt), list(yp), list(s))
        df = auditor.fairness_metrics()
        assert not df.empty


class TestFairnessAuditorPairwise:
    def _data(self):
        rng = np.random.RandomState(11)
        n = 200
        return (
            rng.randint(0, 2, size=n),
            rng.randint(0, 2, size=n),
            rng.choice(["A", "B", "C"], size=n),
        )

    def test_pair_excludes_other_groups(self):
        y_true, y_pred, sens = self._data()
        aud = FairnessAuditor(y_true, y_pred, sens,
                              priv_group="A", unpriv_group="B")
        keep = (sens == "A") | (sens == "B")
        assert len(aud.y_true) == keep.sum()
        manual = FairnessAuditor(y_true[keep], y_pred[keep],
                                 (sens[keep] == "A").astype(int))
        pd.testing.assert_frame_equal(
            aud.fairness_metrics(), manual.fairness_metrics()
        )

    def test_pair_same_value_raises(self):
        y_true, y_pred, sens = self._data()
        with pytest.raises(ValueError, match="must differ"):
            FairnessAuditor(y_true, y_pred, sens,
                            priv_group="A", unpriv_group="A")

    def test_default_is_priv_vs_rest(self):
        y_true, y_pred, sens = self._data()
        aud = FairnessAuditor(y_true, y_pred, sens, priv_group="A")
        assert len(aud.y_true) == len(y_true)
        assert set(np.unique(aud._sens_binary)) <= {0, 1}
