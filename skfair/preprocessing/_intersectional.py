"""Intersectional binariser for multi-attribute privilege definitions."""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class IntersectionalBinarizer(BaseEstimator, TransformerMixin):
    """
    Creates a single binary Protected Group feature (1=Privileged, 0=Unprivileged)
    from complex, user-defined intersectional criteria.

    Privilege is defined by an OR condition over a list of AND conditions (rules).
    Supports equality, list inclusion, and threshold operators (>, <, >=, <=, !=).

    Parameters
    ----------
    privileged_definition : dict or list of dicts
        Defines the criteria for the privileged group.
        - Simple (AND): ``{"race": "White", "sex": "Male"}``
        - Complex (OR of ANDs): ``[{"race": "White", "sex": "Male"},
          {"age": {">": 65}}]``

    group_col_name : str, default="_is_privileged"
        The name of the new binary column to be created.

    privileged_value : int or float, default=1
        The value representing the privileged group.
    """

    def __init__(self, privileged_definition=None, group_col_name="_is_privileged", privileged_value=1):
        if privileged_definition is None:
            raise ValueError("privileged_definition must be provided.")

        self.privileged_definition = privileged_definition
        self.group_col_name = group_col_name
        self.privileged_value = privileged_value

        # Standardize rules to a list (OR over rules, AND within each rule)
        if isinstance(self.privileged_definition, dict):
            self._rules = [self.privileged_definition]
        elif isinstance(self.privileged_definition, list):
            self._rules = self.privileged_definition
        else:
            raise ValueError(
                "privileged_definition must be a dict (for a single AND rule) "
                "or a list of dicts (for OR of AND rules)."
            )

    def fit(self, X, y=None):
        """No fitting necessary for this transformation (stateless)."""
        # We might still want to validate type here for early failure.
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"IntersectionalBinarizer.fit expects X to be a pandas DataFrame, "
                f"got {type(X).__name__} instead."
            )
        return self

    def transform(self, X, y=None):
        """
        Applies the intersectional rules to create the binary feature.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features containing all sensitive attributes referenced in privileged_definition.

        Returns
        -------
        X_out : pandas.DataFrame
            Copy of X with an additional binary column `group_col_name`.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"IntersectionalBinarizer.transform expects X to be a pandas DataFrame, "
                f"got {type(X).__name__} instead."
            )

        X_df = X.copy()

        # Initialize final mask: False = unprivileged by default
        final_privilege_mask = pd.Series(False, index=X_df.index)

        # OR over rule dictionaries
        for rule_dict in self._rules:
            # AND within a single rule
            rule_mask = pd.Series(True, index=X_df.index)

            for attribute, value in rule_dict.items():
                if attribute not in X_df.columns:
                    raise KeyError(
                        f"Protected attribute '{attribute}' not found in the input data columns."
                    )

                # Threshold / operator-based rule: {"age": {">": 65}}
                if isinstance(value, dict):
                    if len(value) != 1:
                        raise ValueError(
                            f"Operator dictionary for '{attribute}' must contain exactly one key (operator)."
                        )

                    operator, threshold = next(iter(value.items()))

                    if operator == ">":
                        rule_mask &= X_df[attribute] > threshold
                    elif operator == "<":
                        rule_mask &= X_df[attribute] < threshold
                    elif operator == ">=":
                        rule_mask &= X_df[attribute] >= threshold
                    elif operator == "<=":
                        rule_mask &= X_df[attribute] <= threshold
                    elif operator == "!=":
                        rule_mask &= X_df[attribute] != threshold
                    else:
                        raise ValueError(
                            f"Unsupported operator '{operator}' used for attribute '{attribute}'."
                        )

                # List inclusion: {"race": ["White", "Asian"]}
                elif isinstance(value, list):
                    rule_mask &= X_df[attribute].isin(value)

                # Equality: {"sex": "Male"}
                else:
                    rule_mask &= X_df[attribute] == value

            # OR into the final mask
            final_privilege_mask |= rule_mask

        # Add binary column
        X_df[self.group_col_name] = np.where(
            final_privilege_mask,
            self.privileged_value,
            0,  # value for the unprivileged group
        )

        return X_df
