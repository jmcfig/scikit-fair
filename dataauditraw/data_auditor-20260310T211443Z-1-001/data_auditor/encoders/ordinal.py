import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..profiler import DataProfiler


class ProfilerOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, profiler:DataProfiler=None, columns=None, categories=None, handle_unknown="use_nan"):
        """Perform ordinal encoding.
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
        
        Parameters
        ----------
        profiler : DataProfiler, optional
            An instance of DataProfiler, for saving mappings.
            Optional so that it works standalone too.
        columns : list of str, optional
            Columns to apply the encoding to. If None, encodes all columns.
            Use columns for flexible control (standalone pipelines).
        categories: list of list, optional
            Defined the expected ordered categories for the ith column.
        handle_unknown : {"use_nan", "error"}
            How to handle unknown categories during transform.
        """
        self.profiler = profiler
        self.columns = columns
        self.categories = categories
        self.handle_unknown = handle_unknown

        # provide scaffolding to save mappings during fit()
        self.mappings = {} # {col: {label: int}}
        self.inverse_mappings = {} # {col: {int: label}}

    def get_encoder_type(self):
        return "Ordinal Encoder"

    def fit(self, X, y=None):
        """Scan columns and build mappings."""
        X = pd.DataFrame(X)

        cols_to_map = self.columns if self.columns else list(X.columns)

        for i, col in enumerate(cols_to_map):
            if self.categories and i < len(self.categories):
                ordered_values = self.categories[i]
            else:
                ordered_values = pd.Series(X[col].dropna().unique()).sort_values().tolist() # deterministic encoding
            
            mapping = {val: idx for idx, val in enumerate(ordered_values)}
            inverse_mapping = {idx: val for val, idx in mapping.items()}
        
            self.mappings[col] = mapping
            self.inverse_mappings[col] = inverse_mapping

        # Update profiler
        if self.profiler is not None:
            self.profiler._set_feature_mappings(map=self.mappings, inverse_map=self.inverse_mappings)
        
        return self # always return self with a fit()
    
    def transform(self, X, y=None):
        """Apply mappings to turn labels into numerical values."""
        X = pd.DataFrame(X).copy()

        cols_to_map = self.columns if self.columns else list(X.columns)
        
        for col in cols_to_map:
            if col not in self.mappings:
                raise KeyError(f"Error: No mapping found for column: {col}")
            
            if ProfilerOrdinalEncoder._value_not_in_mappings(X, col, self.mappings):
                if self.handle_unknown == "error":
                    raise ValueError(f"Error: Category not in mappings: {col}")
                elif self.handle_unknown != "use_nan":
                    raise ValueError(f"Error: Invalid handle_unknown option: {self.handle_unknown}")
                
            X[col] = X[col].map(self.mappings[col]) # this will type-cast to float if there are NaN
    
        return X
        
    def inverse_transform(self, X, y=None):
        """Reverse integers back to original labels."""
        X = pd.DataFrame(X).copy()

        cols_to_map = self.columns if self.columns else list(X.columns)
        
        for col in cols_to_map:
            if col not in self.inverse_mappings:
                raise KeyError(f"Error: No inverse mapping found for column: {col}")
            
            if ProfilerOrdinalEncoder._value_not_in_mappings(X, col, self.inverse_mappings):
                if self.handle_unknown == "error":
                    raise ValueError(f"Error: Category not in mappings: {col}")
                elif self.handle_unknown != "use_nan":
                    raise ValueError(f"Error: Invalid handle_unknown option: {self.handle_unknown}")
                
            X[col] = X[col].map(self.inverse_mappings[col])
    
        return X

    @staticmethod
    def _value_not_in_mappings(df, col:str, mappings:list):
        """Checks whether the values of a column exist in mappings.
        Supports either mapping direction.
        """
        for category in list(df[col].dropna().unique()):
            if category not in list(mappings[col].keys()):
                return True