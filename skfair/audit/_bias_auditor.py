"""Pre-model bias auditor for data-level disparity analysis."""

import numpy as np
import pandas as pd

from ._plots import _plot_bar, _plot_histogram_by_group


class BiasAuditor:
    """Analyse data-level disparity before modelling.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Binary target labels (0/1).
    sens_attr : str
        Column name of the sensitive attribute in *X*.
    priv_group : int or str, default=1
        Value in *sens_attr* that represents the privileged group.
    pos_label : int or str, default=1
        Value in *y* that represents the favourable outcome.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y,
        sens_attr: str,
        priv_group=1,
        pos_label=1,
    ):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if sens_attr not in X.columns:
            raise ValueError(f"'{sens_attr}' not found in X columns.")

        self.X = X
        self.y = np.asarray(y)
        self.sens_attr = sens_attr
        self.priv_group = priv_group
        self.pos_label = pos_label

        self._group_col = X[sens_attr]

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def group_proportions(self) -> pd.DataFrame:
        """Population share of each group value.

        Returns
        -------
        pd.DataFrame
            Columns: ``count``, ``proportion``.
        """
        counts = self._group_col.value_counts()
        props = counts / counts.sum()
        return pd.DataFrame({"count": counts, "proportion": props})

    def target_rate_by_group(self) -> pd.DataFrame:
        """Positive-outcome rate for each group value.

        Returns
        -------
        pd.DataFrame
            Columns: ``count``, ``positive_rate``.
        """
        df = pd.DataFrame({
            self.sens_attr: self._group_col.values,
            "target": self.y,
        })
        grouped = df.groupby(self.sens_attr)["target"]
        counts = grouped.count()
        rates = grouped.apply(lambda s: (s == self.pos_label).mean())
        return pd.DataFrame({"count": counts, "positive_rate": rates})

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_group_proportions(self, **kwargs) -> tuple:
        """Bar chart of group proportions.

        Returns
        -------
        (fig, ax)
        """
        props = self.group_proportions()["proportion"]
        return _plot_bar(
            props,
            title=f"Group Proportions ({self.sens_attr})",
            xlabel=self.sens_attr,
            ylabel="Proportion",
            **kwargs,
        )

    def plot_target_rates(self, **kwargs) -> tuple:
        """Bar chart of positive-outcome rate per group.

        Returns
        -------
        (fig, ax)
        """
        rates = self.target_rate_by_group()["positive_rate"]
        return _plot_bar(
            rates,
            title=f"Positive Outcome Rate by {self.sens_attr}",
            xlabel=self.sens_attr,
            ylabel="Rate",
            **kwargs,
        )

    def plot_feature_distribution(self, feature: str, **kwargs) -> tuple:
        """Histogram of *feature* split by the sensitive attribute.

        Parameters
        ----------
        feature : str
            Column name in *X* to visualise.

        Returns
        -------
        (fig, ax)
        """
        if feature not in self.X.columns:
            raise ValueError(f"'{feature}' not found in X columns.")

        return _plot_histogram_by_group(
            self.X[feature],
            self._group_col.values,
            feature_name=feature,
            group_name=self.sens_attr,
            **kwargs,
        )

    def plot_summary(self, features=None) -> list:
        """Display all bias plots at once.

        Parameters
        ----------
        features : list of str, optional
            Features to plot distributions for.  Defaults to all numeric
            columns in *X* excluding the sensitive attribute.

        Returns
        -------
        list of (fig, ax)
        """
        figs = [
            self.plot_group_proportions(),
            self.plot_target_rates(),
        ]

        if features is None:
            features = [
                c for c in self.X.select_dtypes("number").columns
                if c != self.sens_attr
            ]

        for feat in features:
            figs.append(self.plot_feature_distribution(feat))

        return figs
