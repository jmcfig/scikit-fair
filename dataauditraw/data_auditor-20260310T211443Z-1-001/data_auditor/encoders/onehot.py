import pandas as pd
import numpy as np
from ..profiler import DataProfiler
from sklearn.base import BaseEstimator, TransformerMixin
import copy


class ProfilerOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, profiler:DataProfiler=None, columns=None, drop=None, handle_unknown="use_nan"):
        """Perform One-Hot-Encoding
        
        Parameters
        ----------
        profiler : DataProfiler, optional
            An instance of DataProfiler, for saving mappings.
            Optional so that it works standalone too.
        columns : list of str, optional
            Columns to apply the encoding to. If None, encodes all columns.
            Use columns for flexible control (standalone pipelines).
        drop : {"if_binary", "last"}
            How to drop one of the categories per feature. 
            "if_binary" removes 1 category from binary columns.
            "last" keeps only n-1 columns for nominal data.
        handle_unknown : {"use_nan", "error"}
            How to handle unknown categories during transform.
        """
        self.profiler = profiler
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.drop = drop

        self.mappings = {} # {col: [cat1, cat2, cat3]}, e.g., {'Gender': ['F', 'M']}
        self.inverse_mappings = {}

    def get_encoder_type(self):
        return "OHE"
    
    def fit(self, X, y=None):
        """Scan columns and build mappings."""
        X = pd.DataFrame(X)

        cols_to_map = self.columns if self.columns else list(X.columns)
    
        for col in cols_to_map:
            categories = pd.Series(X[col].dropna().unique()).sort_values().tolist() # deterministic encoding
            self.mappings[col] = categories     # save to rebuild at transform time

            # pre-save inverse mappings
            for category in categories:
                onehot_col = f"{col}_{category}"
                self.inverse_mappings[onehot_col] = (col, category)
        
        # Update profiler
        if self.profiler is not None:
            self.profiler._set_feature_mappings(map=self.mappings, inverse_map=self.inverse_mappings)
        
        return self
    
    def transform(self, X, y=None):
        """Applies mappings to create new one hot columns."""
        X = pd.DataFrame(X).copy()
        
        cols_to_map = self.columns if self.columns else list(X.columns)
    
        for col in cols_to_map:
            if col not in self.mappings:
                raise KeyError(f"Error: No mapping found for column: {col}")
                        
            if ProfilerOneHotEncoder._value_not_in_mappings(X, col, self.mappings):
                if self.handle_unknown == "error":
                    raise ValueError(f"Error: Category not in mappings: {col}")
                elif self.handle_unknown != "use_nan":
                    raise ValueError(f"Error: Invalid handle_unknown option: {self.handle_unknown}")

            X[col] = X[col].where(X[col].isin(self.mappings[col]))

            cats_to_encode = self.mappings[col][:] # shallow copy, new list with same els
            
            if self.drop is not None:
                if self.drop == "if_binary":
                    if len(cats_to_encode) == 2:
                        del cats_to_encode[-1] # drop 1 if binary
                elif self.drop == "last":
                    if len(cats_to_encode) >= 2:
                        del cats_to_encode[-1] # drop last category, keep n-1
                else:
                    raise ValueError(f"Error: Invalid drop option: {self.drop}")

            for cat in cats_to_encode:
                ohe_col = f"{col}_{cat}"
                X[ohe_col] = ((X[col] == cat).astype(int)).where(X[col].notnull(), other=np.nan)
                
        # drop original columns
        X.drop(columns=cols_to_map, inplace=True)
        return X
    
    def inverse_transform(self, X, y=None):
        """Reverse one hot vectors to original categories."""
        X = pd.DataFrame(X).copy()

        col_groups = {}
        
        # Go through inverse_mappings and group ohe columns
        # inverse_mappings: {'Gender_F': ('Gender', 'F'), 'Gender_M': ('Gender', 'M')}
        # col_groups = {'Gender': [('Gender_F', 'F'), ('Gender_M', 'M')]}
        for ohe_col, (orig_col, cat) in self.inverse_mappings.items():
            if orig_col not in col_groups.keys():
                col_groups[orig_col] = []
            col_groups[orig_col].append((ohe_col, cat))
            
        result = pd.DataFrame(index=X.index) # empty dataframe with same index (row labels) as X

        for orig_col in col_groups.keys():
            result[orig_col] = X.apply(
                ProfilerOneHotEncoder.decode_row, 
                axis=1,
                args=(col_groups[orig_col], self.drop, self.mappings[orig_col])
            )

        ohe_cols = list(self.inverse_mappings.keys())
        X.drop(columns=[col for col in ohe_cols if col in X.columns], inplace=True)
        
        X = pd.concat([X, result], axis=1)

        return X

    @staticmethod
    def _value_not_in_mappings(df, col:str, mappings:list):
        """Checks whether the values of a column exist in mappings.
        Supports either mapping direction.
        """
        for category in list(df[col].dropna().unique()):
            if category not in mappings[col]:
                return True

    @staticmethod
    def decode_row(row, ohe_list, drop, mappings):
        for ohe_col, category in ohe_list:
            if ohe_col in row:
                if row[ohe_col] == 1: return category
                elif pd.isna(row[ohe_col]): return np.nan
        if drop in ["if_binary", "last"]:
            return mappings[-1]
        return np.nan