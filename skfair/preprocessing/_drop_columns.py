"""DropColumns transformer for removing columns in sklearn pipelines."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DropColumns(BaseEstimator, TransformerMixin):
    """
    Drop specified columns from a DataFrame in a sklearn pipeline.

    Useful for removing sensitive attributes before an estimator while
    keeping them available earlier in the pipeline for fairness
    preprocessing or evaluation.

    Parameters
    ----------
    columns : str or list of str
        Column name(s) to drop.

    Examples
    --------
    >>> from skfair.preprocessing import DropColumns
    >>> from skfair.datasets import load_adult
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_adult()
    >>> pipe = Pipeline([
    ...     ("drop_sensitive", DropColumns("sex")),
    ...     ("clf", LogisticRegression()),
    ... ])
    >>> pipe.fit(X, y)  # doctest: +SKIP
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        """Record which columns exist and should be dropped.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y : ignored

        Returns
        -------
        self
        """
        cols = [self.columns] if isinstance(self.columns, str) else list(self.columns)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in X: {missing}")
        self._cols_to_drop = cols
        return self

    def transform(self, X, y=None):
        """Drop the specified columns from X.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y : ignored

        Returns
        -------
        X_transformed : pd.DataFrame
            DataFrame with the specified columns removed.
        """
        check_is_fitted(self, "_cols_to_drop")
        return X.drop(columns=self._cols_to_drop)
