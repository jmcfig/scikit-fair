"""Post-model fairness auditor for prediction-level analysis."""

import numpy as np
import pandas as pd

from ..metrics._fairness import _split_by_group
from ..metrics._performance import (
    accuracy,
    false_negative_rate,
    false_positive_rate,
    true_negative_rate,
    true_positive_rate,
)
from ..metrics._fairness import (
    accuracy_parity,
    average_odds_difference,
    disparate_impact,
    equal_opportunity_difference,
    equal_opportunity_ratio,
    false_negative_rate_difference,
    predictive_equality,
    statistical_parity_difference,
    true_negative_rate_difference,
)
from ._plots import _plot_grouped_bars, _plot_metric_bars, _plot_radar


class FairnessAuditor:
    """Audit fairness of model predictions across groups.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0/1).
    y_pred : array-like
        Predicted binary labels (0/1).
    sens_attr : array-like
        Binary group indicator aligned with *y_true* / *y_pred*.
        The privileged group is identified by *priv_group*.
    priv_group : int or str, default=1
        Value in *sens_attr* that represents the privileged group.
    pos_label : int or str, default=1
        Value that represents the favourable outcome.
    """

    def __init__(
        self,
        y_true,
        y_pred,
        sens_attr,
        priv_group=1,
        pos_label=1,
    ):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.sens_attr = np.asarray(sens_attr)
        self.priv_group = priv_group
        self.pos_label = pos_label

        # Pre-compute binary mask expected by skfair.metrics
        self._sens_binary = (self.sens_attr == self.priv_group).astype(int)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def performance_by_group(self) -> pd.DataFrame:
        """Per-group performance metrics.

        Returns
        -------
        pd.DataFrame
            Rows = metric names, columns = ``unprivileged``, ``privileged``.
        """
        (yt_u, yp_u), (yt_p, yp_p) = _split_by_group(
            self.y_true, self.y_pred, self._sens_binary,
        )

        metric_fns = {
            "Accuracy": accuracy,
            "TPR": true_positive_rate,
            "FPR": false_positive_rate,
            "TNR": true_negative_rate,
            "FNR": false_negative_rate,
        }

        rows = {}
        for name, fn in metric_fns.items():
            rows[name] = {
                "unprivileged": fn(yt_u, yp_u),
                "privileged": fn(yt_p, yp_p),
            }

        return pd.DataFrame(rows).T

    def fairness_metrics(self) -> pd.DataFrame:
        """Compute all fairness metrics.

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame (``value``) indexed by metric name.
        """
        s = self._sens_binary
        yt, yp = self.y_true, self.y_pred

        results = {
            "Disparate Impact": disparate_impact(yt, yp, s),
            "Statistical Parity Diff": statistical_parity_difference(yt, yp, s),
            "Equal Opportunity Diff": equal_opportunity_difference(yt, yp, s),
            "Equal Opportunity Ratio": equal_opportunity_ratio(yt, yp, s),
            "Average Odds Diff": average_odds_difference(yt, yp, s),
            "TNR Difference": true_negative_rate_difference(yt, yp, s),
            "FNR Difference": false_negative_rate_difference(yt, yp, s),
            "Predictive Equality": predictive_equality(yt, yp, s),
            "Accuracy Parity": accuracy_parity(yt, yp, s),
        }

        return pd.DataFrame.from_dict(results, orient="index", columns=["value"])

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_performance_by_group(self, **kwargs) -> tuple:
        """Grouped bar chart of per-group performance metrics.

        Returns
        -------
        (fig, ax)
        """
        df = self.performance_by_group()
        return _plot_grouped_bars(
            df,
            title="Performance by Group",
            ylabel="Score",
            **kwargs,
        )

    def plot_fairness_metrics(self, **kwargs) -> tuple:
        """Horizontal bar chart with colour-coded fairness metrics.

        Returns
        -------
        (fig, ax)
        """
        fm = self.fairness_metrics()["value"]
        return _plot_metric_bars(fm, **kwargs)

    def plot_fairness_radar(self, **kwargs) -> tuple:
        """Radar (spider) chart of all fairness metrics.

        Returns
        -------
        (fig, ax)
        """
        fm = self.fairness_metrics()["value"]
        return _plot_radar(fm, **kwargs)

    def plot_summary(self) -> list:
        """Display all fairness plots at once.

        Returns
        -------
        list of (fig, ax)
        """
        return [
            self.plot_performance_by_group(),
            self.plot_fairness_metrics(),
            self.plot_fairness_radar(),
        ]
