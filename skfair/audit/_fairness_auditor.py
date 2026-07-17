"""Post-model fairness auditor for prediction-level analysis."""

import numpy as np
import pandas as pd

from ..metrics import METRICS
from ..metrics._fairness import _split_by_group
from ..metrics._registry import spec as _metric_spec
from ..metrics._performance import (
    accuracy,
    false_negative_rate,
    false_positive_rate,
    true_negative_rate,
    true_positive_rate,
)
from ._plots import _plot_grouped_bars, _plot_metric_bars, _plot_radar, _split_metrics_by_type


class FairnessAuditor:
    """Audit fairness of model predictions across groups.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0/1).
    y_pred : array-like
        Predicted binary labels (0/1).
    sens_attr : array-like
        Group indicator aligned with *y_true* / *y_pred*; may hold any
        values. The privileged group is identified by *priv_group*.
    priv_group : int or str, default=1
        Value in *sens_attr* that represents the privileged group. All
        remaining values form the unprivileged group, unless
        *unpriv_group* is given.
    unpriv_group : int or str, optional
        Value in *sens_attr* treated as the unprivileged group. When set,
        samples belonging to neither group are excluded from the audit,
        so any pair of groups of a multi-valued attribute can be compared.
    pos_label : int or str, default=1
        Value that represents the favourable outcome.
    """

    def __init__(
        self,
        y_true,
        y_pred,
        sens_attr,
        priv_group=1,
        unpriv_group=None,
        pos_label=1,
    ):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.sens_attr = np.asarray(sens_attr)
        self.priv_group = priv_group
        self.unpriv_group = unpriv_group
        self.pos_label = pos_label

        if unpriv_group is not None:
            if unpriv_group == priv_group:
                raise ValueError("priv_group and unpriv_group must differ.")
            keep = (self.sens_attr == priv_group) | (self.sens_attr == unpriv_group)
            if not keep.any():
                raise ValueError(
                    "No samples belong to either priv_group or unpriv_group."
                )
            self.y_true = self.y_true[keep]
            self.y_pred = self.y_pred[keep]
            self.sens_attr = self.sens_attr[keep]

        # Pre-compute binary mask expected by skfair.metrics
        self._sens_binary = (self.sens_attr == self.priv_group).astype(int)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def performance_by_group(self, metrics=None) -> pd.DataFrame:
        """Per-group performance metrics.

        Parameters
        ----------
        metrics : iterable of str, optional
            Performance metrics to compute, as canonical registry names or
            aliases (e.g. ``["accuracy", "precision", "recall"]``). Rows are
            labelled with each metric's display name, in the given order.
            If None, uses accuracy plus the four confusion-matrix rates.

        Returns
        -------
        pd.DataFrame
            Rows = metric names, columns = ``unprivileged``, ``privileged``.
        """
        (yt_u, yp_u), (yt_p, yp_p) = _split_by_group(
            self.y_true, self.y_pred, self._sens_binary,
        )

        if metrics is None:
            metric_fns = {
                "Accuracy": accuracy,
                "TPR": true_positive_rate,
                "FPR": false_positive_rate,
                "TNR": true_negative_rate,
                "FNR": false_negative_rate,
            }
        else:
            metric_fns = {}
            for name in metrics:
                s = _metric_spec(name)
                if s.kind != "performance":
                    raise ValueError(
                        f"'{name}' is not a performance metric; "
                        f"performance_by_group only accepts performance metrics."
                    )
                metric_fns[s.display] = s.func

        rows = {}
        for name, fn in metric_fns.items():
            rows[name] = {
                "unprivileged": fn(yt_u, yp_u),
                "privileged": fn(yt_p, yp_p),
            }

        return pd.DataFrame(rows).T

    def fairness_metrics(self, metrics=None) -> pd.DataFrame:
        """Compute fairness metrics.

        By default covers every fairness metric in the registry
        (:data:`skfair.metrics.METRICS`) — independence, separation,
        sufficiency and accuracy families — labelled by each metric's display
        name.

        Parameters
        ----------
        metrics : iterable of str, optional
            Fairness metrics to compute, as canonical registry names or
            aliases (e.g. ``["spd", "equal_opportunity_difference"]``),
            reported in the given order. If None, computes the full registry.

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame (``value``) indexed by metric name.
        """
        s = self._sens_binary
        yt, yp = self.y_true, self.y_pred

        if metrics is None:
            specs = list(METRICS.values())
        else:
            specs = []
            for name in metrics:
                sp = _metric_spec(name)
                if sp.kind != "fairness":
                    raise ValueError(
                        f"'{name}' is not a fairness metric; "
                        f"fairness_metrics only accepts fairness metrics."
                    )
                specs.append(sp)

        results = {sp.display: sp.func(yt, yp, s) for sp in specs}

        return pd.DataFrame.from_dict(results, orient="index", columns=["value"])

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_performance_by_group(self, metrics=None, **kwargs) -> tuple:
        """Grouped bar chart of per-group performance metrics.

        Parameters
        ----------
        metrics : iterable of str, optional
            Performance metrics to show (canonical names or aliases);
            forwarded to :meth:`performance_by_group`.

        Returns
        -------
        (fig, ax)
        """
        df = self.performance_by_group(metrics=metrics)
        return _plot_grouped_bars(
            df,
            title="Performance by Group",
            ylabel="Score",
            **kwargs,
        )

    def plot_fairness_metrics(self, metrics=None, **kwargs) -> tuple:
        """Horizontal bar chart with colour-coded fairness metrics.

        Parameters
        ----------
        metrics : iterable of str, optional
            Fairness metrics to show (canonical names or aliases);
            forwarded to :meth:`fairness_metrics`. If None, shows all.
        fair_threshold : float, optional
            Distance from ideal for green (default 0.1).
        warning_threshold : float, optional
            Distance from ideal for orange (default 0.2).
        **kwargs
            Forwarded to :func:`_plot_metric_bars`.

        Returns
        -------
        (fig, ax)
        """
        fm = self.fairness_metrics(metrics=metrics)["value"]
        return _plot_metric_bars(fm, **kwargs)

    def plot_fairness_radar(self, mode="ratio", **kwargs) -> tuple:
        """Radar (spider) chart of fairness metrics.

        Parameters
        ----------
        mode : {"ratio", "difference", "all"}, default="ratio"
            Which metric subset to plot:
            ``"ratio"`` for ratio-based (ideal = 1),
            ``"difference"`` for difference-based (ideal = 0),
            ``"all"`` for every metric.

            The default is ``"ratio"`` because ratio metrics share a common
            ideal of 1, making the radar shape directly interpretable.
            Difference metrics (ideal = 0) collapse toward the centre and
            can go negative, which distorts the polar plot.

            Values are normalised for the radar: ratio metrics use
            ``min(v, 1/v)`` so over- and under-representation are
            symmetric; difference metrics use absolute values.

        Returns
        -------
        (fig, ax)
        """
        fm = self.fairness_metrics()["value"]
        if mode == "all":
            return _plot_radar(fm, title="Fairness Radar (ideal = 1)", **kwargs)
        diff, ratio = _split_metrics_by_type(fm)
        if mode == "ratio":
            return _plot_radar(ratio, title="Fairness Radar — Ratio (ideal = 1)", **kwargs)
        return _plot_radar(diff, title="Fairness Radar — Difference (ideal = 0)", **kwargs)

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
