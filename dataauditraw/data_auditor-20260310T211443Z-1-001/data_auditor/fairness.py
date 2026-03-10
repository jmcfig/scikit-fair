import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from .profiler import DataProfiler
from itertools import combinations
from typing import Union
from sklearn.metrics import confusion_matrix, classification_report


class FairnessEngine:
    def __init__(self, df, sensitive_features:list[str], target:str=None):
        self.df = df
        #TODO: maybe check if these features exist in df and if they are categorical
        self.sensitive_features = sensitive_features
        self.target = target
        self.profiler = DataProfiler(df=df, target=target)
        self.sensitive_groups = FairnessEngine.intersectional_fairness_groups(sensitive_features)
        self.fairness_df = FairnessEngine.create_fairness_df(df=self.df, 
                                                                      sensitive_groups=self.sensitive_groups)
        self.fairness_profiler = DataProfiler(df=self.fairness_df, target=target)
     
    def group_proportions_table(self, group:str, plot_tab=False, fig_size=None) -> pd.DataFrame: 
        """Creates a table of group distribution.
        Group can be a combination of sensitive features, given by
        self.sensitive_groups.
        """
        if group in self.sensitive_groups.keys():
            sensitive_features = self.sensitive_groups[group]
            rows_index = sensitive_features[0] if len(sensitive_features) > 1 else None
            cols = sensitive_features[1:] if rows_index else sensitive_features
            
        pivot = pd.pivot_table(
            self.df,
            index=rows_index,
            columns=cols,
            aggfunc="size",
            fill_value=0
        )

        pivot_percent = round(pivot / pivot.values.sum() * 100, 2)

        if plot_tab:
            if len(sensitive_features) == 1: raise ValueError("Error: Not enough intersection to create heatmap.")
            fig_size = fig_size if fig_size is not None else (max(pivot_percent.shape) * 2, min(pivot_percent.shape) * 2)
            self.profiler.plot_pivot_table(
                index=rows_index,
                columns=cols,
                aggfunc="size", 
                title="Proportions by Group",
                plot_percent=True, 
                figsize=fig_size)

        return pivot_percent

    def target_rate_by_group_table(self, group:str, target:str=None, plot_tab=False, fig_size=None) -> pd.DataFrame:
        """Creates a table of target distribution by group.
        Group can be a combination of sensitive features, given by
        self.sensitive_groups.
        """
        #TODO: add protections for unknown group or target
        target = self.target if target is None else target 
            
        if group in self.sensitive_groups.keys() and target:
            pivot = pd.pivot_table(
                self.df, 
                index=target,
                columns=self.sensitive_groups[group],
                aggfunc="size",
                fill_value=0
            )

            pivot_rate = round(pivot / pivot.values.sum() * 100, 2)

            if plot_tab:
                fig_size = fig_size if fig_size is not None else (max(pivot_rate.shape) * 2, min(pivot_rate.shape) * 2)
                self.profiler.plot_pivot_table(
                    index=target,
                    columns=self.sensitive_groups[group],
                    aggfunc="size", 
                    title="Proportions by Group",
                    plot_percent=True, 
                    figsize=fig_size
                )

            return pivot_rate

    def missing_data_disparity_imbalance(self):
        """Calculates the disparity (imbalance ratio) by feature
        for each group or group aggregation. 
        Returns a dictionary for all groups and all features.
        """
        pass

    def plot_sensitive_groups(self, groups:Union[str, list], target=None, subplot_size=(9, 4)):
        """Plots the distribution of sensitive groups. If target is given, 
        the distribution will be color-coded. Parameter groups can be a str of specific group,
        a list of groups, or "all" (in this case it plots all sensitive groups).
        """
    
        all_groups = self.get_sensitive_groups()

        if groups == "all": groups = all_groups
        elif isinstance(groups, str): groups=[groups]
        elif not (isinstance(groups, list) and all(g in all_groups for g in groups)): 
             raise ValueError("Error: Unknown group passed as argument.")
        
        self.fairness_profiler.plot_mult_histogram(
                col_names=groups,
                hue=target,
                cols=1,
                subplot_size=subplot_size,
        )

    def plot_features_by_group(self, col_names:list, groups:Union[str, list], subplot_size=(9, 4)):
        
        all_groups = self.get_sensitive_groups()

        if groups == "all": groups = all_groups
        elif isinstance(groups, str): groups=[groups]
        elif not (isinstance(groups, list) and all(g in all_groups for g in groups)): 
             raise ValueError("Error: Unknown group passed as argument.")
        
        if isinstance(col_names, str): col_names=[col_names]

        for group in groups:
            self.fairness_profiler.plot_mult_histogram(col_names=col_names, hue=group)

    def get_sensitive_groups(self):
        return list(self.sensitive_groups.keys())
    
    def get_sensitive_features(self):
        return self.sensitive_features


    @staticmethod
    def intersectional_fairness_groups(sensitive_features:list[str]) -> dict:
        """Determines interseccional sensitive groups.
        For each sensitive group, returns the features that
        create it.
        """
        res = []
        for i in range(1, len(sensitive_features) + 1):
            c = list(combinations(sensitive_features, i))
            res.extend(c)
        group_lst = ["_".join(pairs) for pairs in res]

        intersectional_groups = {}
        for g in group_lst:
            intersectional_groups[g] = g.split("_")
        
        return intersectional_groups
    
    @staticmethod
    def create_fairness_df(df: pd.DataFrame, sensitive_groups: dict):
        """Creates a copy of the original dataframe and adds columns
        corresponding to the combination of sensitive features.
        """
        fairness_df = df.copy()
        for group, features in sensitive_groups.items():
            if len(features) > 1:
                fairness_df[group] = fairness_df[features].agg(lambda x: "_".join(x.dropna().astype(str)) 
                                             if x.notna().all() else np.nan, axis=1)
        return fairness_df

    @staticmethod
    def add_y_predictions(df, y_pred):
        """Adds predictions y_pred to a previous dataframe."""
        df["y_pred"] = y_pred
        return df
    
    @staticmethod
    def split_X_y_dem(df, sensitive_groups:list, target:str):
        """Splits the dataframe df into X (feature), 
        y (predictions), and dem (demographics).
        Demographics are given by sensitive_groups.
        """
        remove_cols = [target] + sensitive_groups
        X = df[[col for col in df.columns if col not in remove_cols]]
        y = df[target]
        dem = df[sensitive_groups] 
        return X, y, dem
    
    @staticmethod
    def compute_cm_metrics(df, sensitive_group:str, sensitive_label:str, target_str:str, ypred_str:str):
        """Computes basic performance metrics for a particular group and label."""

        df = df[df[sensitive_group] == sensitive_label]
        cm = confusion_matrix(df[target_str], df[ypred_str])

        tn, fp, fn, tp = cm.ravel()
        n = tp + tn + fp + fn

        metrics = {
            "acc": (tp + tn) / n,           # accuracy
            "tpr": tp / (tp + fn),          # true positive rate
            "fpr": fp / (fp + tn),          # false positive rate
            "sr": (tp + fp) / n,            # success rate
            "fnr": 1 - tp / (tp + fn)       # false negative rate
            }
        
        metrics["sensitive_group"] = sensitive_group
        metrics["sensitive_label"] = sensitive_label
        
        return metrics

    @staticmethod
    def compute_fairness_metrics(cmp:dict, cmu:dict):
        """Computed standard group fairness metrics. Performance metrics for the 
        priviledged and unpriviledged groups are given as input.
        """
        fairness_metrics = {
            "acc": round(cmu["acc"] / cmp["acc"], 2),           # ACC_0 / ACC_1 (Accuracy Parity)
            "pe": round(cmu["fpr"] / cmp["fpr"], 2),            # Predictive Equality
            "eor_fnr": round(cmu["fnr"] / cmp["fnr"], 2),       # Also Equal Opportunity Ratio
            "sp": round(cmu["sr"] - cmp["sr"], 2),              # Statistical Parity
            "di": round(cmu["sr"] / cmp["sr"], 2),              # Disparate Impact
            "eod": round(cmu["tpr"] - cmp["tpr"], 2),           # Equal Opportunity Difference
            "eor_tpr": round(cmu["tpr"] / cmp["tpr"], 2),       # Equal Opportunity Ratio
            "aod": round(0.5 *                                  # Average Odds Difference
                         (cmu["fpr"] - cmp["fpr"]) + 
                         (cmu["tpr"] - cmp["tpr"]), 2)
            }
        
            # TODO: Add common needed ratios here
        
        fairness_metrics["unpriviledged_group"] = "_".join([cmu["sensitive_group"], cmu["sensitive_label"]])
        fairness_metrics["priviledged_group"] = "_".join([cmp["sensitive_group"], cmp["sensitive_label"]])
        
        return fairness_metrics
    
    @staticmethod
    def print_metrics_report(metrics:dict, print_only:list=None):
        """Pretty prints a performance or fairness report."""
        performance_flag = "sensitive_group" in metrics.keys()
        
        if performance_flag:
            print(f'\033[1mPerformance Results for {metrics["sensitive_group"]}: \033[0m{metrics["sensitive_label"]}')
        else:
            print(f"\033[1mComparing Fairness Results")
            print(f'\033[1mPriviledged Group: \033[0m{metrics["priviledged_group"]}')
            print(f'\033[1mUnpriviledged Group: \033[0m{metrics["unpriviledged_group"]}')
        
        for k, v in metrics.items():
            if k in ("sensitive_group", "sensitive_label", "unpriviledged_group", "priviledged_group"):
                continue
            if print_only is not None:
                skip_cols = list(set(metrics.keys()) - set(print_only))
                if k in skip_cols:
                    continue
            print(f'\033[1m{k.upper()}: \033[0m{v:.2f}')

            