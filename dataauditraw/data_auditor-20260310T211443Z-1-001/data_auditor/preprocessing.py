from .profiler import DataProfiler
from typing import Callable, Optional
from .encoders.ordinal import ProfilerOrdinalEncoder
from .encoders.onehot import ProfilerOneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

# And functions to perform MinMax, Scaling, etc to each of them
# Note sure how Pipelines work and whether we'd like to incorporate them here, probably we won't

# Decorators
def track_changes(func: Callable):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.change_logs.append(f"{func.__name__}")
        return result   # can't be swallowed by decorator to allow chaining
    return wrapper


class DataPreprocessor:
    def __init__(self, profiler: DataProfiler):
        self.profiler = profiler    # shared reference
        #self.df = copy.deepcopy(profiler.df)
        self.df = profiler.df.copy()
        self.encoders = {}
        self.column_transformers = {}
        self.change_logs = []
        self.log_details = []
    
    @track_changes
    def delete_features(self, col_names: list):
        """Delete specific features from the dataset and update profiler."""
        if self.profiler.target in col_names:
            self.profiler.target = None
        self.df.drop(columns=col_names, inplace=True) # modify directly, do not create new dataframe
        self.profiler.df = self.df
        self.profiler._refresh()
        log_msg = f"Deleted: {col_names}"
        self.log_details.append(log_msg)
        return self # allows for method chaining (e.g. processor.delete_features().scale_numericals())
                    # without return self, you need to call separately: 
                    # processor.delete_features()
                    # processor.scale_numericals()
        #TODO: Maybe delete corresponding feature mappings, if feature gets deleted? 
        # (self.mapping and profiler.mappings?)

    @track_changes
    def set_semantic_feature_types(self,
                          continuous_cols: Optional[list]=None, 
                          ordinal_cols: Optional[list]=None, 
                          nominal_cols: Optional[list]=None, 
                          binary_cols: Optional[list]=None):
        """Defines continuous, ordinal, nominal, and binary features."""
        self.profiler._set_semantic_feature_types(
            cont_cols = continuous_cols,
            ord_cols = ordinal_cols,
            nom_cols = nominal_cols,
            bin_cols = binary_cols)
        log_msg = ""
        if continuous_cols: log_msg += f"Continuous: {continuous_cols}, "
        if ordinal_cols: log_msg += f"Ordinal: {ordinal_cols} "
        if nominal_cols: log_msg += f"Nominal: {nominal_cols} "
        if binary_cols: log_msg += f"Binary: {binary_cols} "
        self.log_details.append(log_msg)
        return self
    
    @track_changes
    def change_feature_types(self, num_cols:list[str]=None, cat_cols:list[str]=None):
        self.profiler._override_feature_types(numeric=num_cols, categorical=cat_cols)
        self.profiler._refresh()
        log_msg = ""
        if num_cols: log_msg += f"Num Cols: {num_cols}"
        if cat_cols: log_msg += f"Cat Cols: {cat_cols}"
        self.log_details.append(log_msg)
        return self
    
    def print_log_report(self):
        print("===== Track Changes =====")
        for i in range(len(self.change_logs)):
            print(f"Method: {self.change_logs[i]}. {self.log_details[i]}")

    @track_changes
    def ohe_transform(self, profiler=None, columns:list[str]=None, drop=None, handle_unknown="use_nan"):
        ohe = ProfilerOneHotEncoder(
            profiler=profiler if profiler is not None else self.profiler,
            columns=columns,
            drop=drop,
            handle_unknown=handle_unknown
        )
        
        self.df = ohe.fit_transform(self.df)
        self.profiler.df = self.df
        key = f"OHE_{'_'.join(ohe.columns)}"    
        self.encoders[key] = ohe
        self.profiler._refresh()
        log_msg = f"OHE Encoding: {ohe.columns}"
        self.log_details.append(log_msg)
        return self
    
    @track_changes
    def ordinal_transform(self, profiler=None, columns:list[str]=None, categories=None, handle_unknown="use_nan"):
        oe = ProfilerOrdinalEncoder(
            profiler=profiler if profiler is not None else self.profiler,
            columns=columns,
            categories=categories,
            handle_unknown=handle_unknown
        )
        
        self.df = oe.fit_transform(self.df)
        self.profiler.df = self.df
        key = f"OrdinalEnc_{'_'.join(oe.columns)}"    
        self.encoders[key] = oe
        self.profiler._refresh()
        log_msg = f"Ordinal Encoding: {oe.columns}"
        self.log_details.append(log_msg)
        return self
    
    @track_changes
    def encoder_inverse_transform(self, encoder_key:str):
        encoder = self.encoders[encoder_key]
        self.df = encoder.inverse_transform(self.df)
        self.profiler.df = self.df
        self.profiler._refresh()
        encoder_type = encoder.get_encoder_type()
        log_msg = f"{encoder_type} Inverse Transform: {encoder.columns}"
        self.log_details.append(log_msg)
        return self
    
    @track_changes
    def auto_col_transformer(self):
        """Automatically build ColumnTransformer based on profiler metadata."""

        transformer_constructor = {
            "cont_cols": lambda: ("scale_continuous", MinMaxScaler(), getattr(self.profiler, "cont_cols", [])),
            "ord_cols": lambda: ("encode_ordinal", ProfilerOrdinalEncoder(profiler=self.profiler), getattr(self.profiler, "ord_cols", [])),
            "nom_cols": lambda: ("encode_nominal", ProfilerOneHotEncoder(profiler=self.profiler), getattr(self.profiler, "nom_cols", [])),
            "bin_cols": lambda: ("encode_binary", ProfilerOneHotEncoder(profiler=self.profiler), getattr(self.profiler, "bin_cols", []))
        }

        transformers = []
        log_transformers = ""

        for constructor in transformer_constructor.values():
            name, transformer, cols = constructor() # call lambda function to return tuple
            if cols:
                transformers.append((name, transformer, cols))
                log_transformers += f"{name}"
      
        if not transformers:
            raise Exception("Error: No transformer was created.")
        

        ct = ColumnTransformer(transformers=transformers, remainder="passthrough")
        log_msg = "Automatic Column Transformer Created: " + log_transformers.rstrip("_")
        self.log_details.append(log_msg)
        key = f"ColumnTransformer_{log_transformers.rstrip("_")}"    
        self.column_transformers[key] = ct
        
        return ct
        
    def get_encoders(self):
        return self.encoders

