import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math

class DataProfiler:
    """Organizes metadata and offers visual diagnostics on tabular datasets."""
    def __init__(self, df: pd.DataFrame, target: str = None):
        self.df = df
        self.target = target
        self.analysis_summary = {}
        self.metadata_summary = {}
        self._feature_overrides = {
            "num_cols": set(),
            "cat_cols": set()
        }
        self.cont_cols = None
        self.ord_cols = None
        self.nom_cols = None
        self.bin_cols = None
        self._analyze()
    
    def get_data(self):
        return self.df
    
    @property
    def summary(self):
        return {**self.analysis_summary, **self.metadata_summary}
    
    def print_report(self):
        # TODO: add all, metadata, analysis
        for k,v in self.summary.items():
            print(f"{k}: {v}")

    def plot_histogram(self, x: str, hue=None, kde=None, multiple="layer", shrink=1, title=None, figsize=(6, 4)):
        """Plot basic histogram.
        
        Parameters
        ----------
        x: str
            The name of the column to plot the histogram
        kde: bool, optional
            If True, computes kernel density and shows on the plot
        hue: str, optional
            Vector or key in self.df. Semantic feature to determine the color of plot elements.
        multiple: {"layer", "dodge", "stack", "fill"}
            Approach to resolve multiple elements when semantic mapping creates subsets.
        """
        if multiple == "dodge": shrink = 0.8
        plt.figure(figsize=figsize)
        sns.histplot(data=self.df, x=x, hue=hue, kde=kde, multiple=multiple, shrink=shrink)
        if not title:
            title = f"Histogram of {x}"
            if hue: title += f" based on {hue}"
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_scatter(self, x: str, y: str, hue=None, style=None, title=None, x_legend=None, y_legend=None):
        """Plot basic scatterplot.
        
        Parameters
        ----------
        x: str
            The name of the column on the xx axis
        y: str
            The name of the column on the yy axis
        hue: str, optional
            Grouping column to determine the color of plot elements.
        style: str, optional
            Grouping column that will produce different markers.
        
        """
        plt.figure(figsize=(6, 4))
        palette = "deep" if hue else None
        sns.scatterplot(self.df, x=x, y=y, hue=hue, style=style, palette=palette)
        plt.xlabel(x_legend if x_legend else x)
        plt.ylabel(y_legend if y_legend else y)
        text_title = f"{x} versus {y}"
        if hue: text_title += f" by {hue}"
        plt.title(title if title else f"{x} versus {y}")
        plt.tight_layout()
        plt.show()

    def plot_pivot_table(self, index: str, columns: str, aggfunc="mean", title:str=None, values:str=None, figsize:tuple=(6,4), plot_percent=False):
        """Plot heatmap based on pivot table.

        Parameters
        ----------
        values: str
            Column to aggregate.
        index: str
            Keys to group by on the pivot table.
        columns: str
            Keys to group by on the pivot table.
        aggfunc: {"mean", "max", "min"}
            Function to calculate aggregates.
        """
        fill_value = 0 if aggfunc == "size" else None

        pivot_df = self.df.pivot_table(
            values=values,
            index=index,
            columns=columns,
            aggfunc = aggfunc,
            fill_value=fill_value
        )

        if aggfunc == "size" and plot_percent:
            pivot_df = round(pivot_df / pivot_df.values.sum() * 100, 2)

        plt.figure(figsize=figsize)

        label = f"{aggfunc}_{values}" if values else f"{aggfunc}"
        ax = sns.heatmap(pivot_df,
                         annot=True,
                         fmt=".1f",
                         linewidth=.5,
                         cbar_kws={"label": label})
        ax.set(xlabel=columns, ylabel=index)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        plt.title(title if title else f"{values if values else aggfunc} by {index} and {columns}")
        plt.show()

    def plot_mult_histogram(self, col_names: list, hue=None, cols=2, subplot_size=(6, 4)):
        """Plot multiple histograms in a grid layout.

        Parameters
        ----------
        col_names: list
            List of column names to plot.
        hue: str, optional
            Column used to group histograms.
        cols: int
            Number of columns in the grid layout.
        figsize: tuple
            Size of the figure.
        """
        n = len(col_names)
        rows = math.ceil(n/cols)

        fig_width, fig_height = subplot_size[0] * cols, subplot_size[1] * rows 
        
        _ , axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        axes = np.atleast_1d(axes)
        axes = axes.flatten()

        for i, col in enumerate(col_names):
            ax = axes[i]
            if col in self.cat_cols: sns.histplot(self.df, x=col, hue=hue, multiple="dodge", shrink=0.8, ax=ax)
            else: sns.histplot(self.df, x=col, hue=hue, ax=ax)
            title_text = f"{col}"
            if hue: title_text += f" by {hue}"
            ax.set_title(title_text)

        # Remove extra axis 
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_mult_scatter(self, col_names: list, fixed_col=None, hue=None, style=None, cols=2, figsize=(12, 4)):
        """Plot all scatter, by hue, using numerical features.

        Parameters
        ----------
        col_names: list
            List of column names to plot.
        hue: str, optional
            Column used to group histograms.
        cols: int
            Number of columns in the grid layout.
        figsize: tuple
            Size of the figure.
        """
        if fixed_col: 
            combinations = set()
            col_names.remove(fixed_col)
            for i in col_names:
                combinations.add((fixed_col, i))
        else: combinations = DataProfiler.combinatorial_k_2(col_names)

        n = len(combinations)
        rows = math.ceil(n/cols)
        palette = "deep" if hue else None

        _ , axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1]*rows))
        axes = axes.flatten()

        for i, combo in enumerate(combinations):
            ax = axes[i]
            if combo[0] in self.num_cols and combo[1] in self.num_cols:
                sns.scatterplot(self.df, x=combo[0], y=combo[1], hue=hue, style=style, palette=palette, ax=ax)
                title_text = f"{combo[0]} versus {combo[1]}"
                if hue: title_text += f" by {hue}"
                ax.set_title(title_text)
            else: raise ValueError("One or more features is not quantitative.") # TODO: raise specific error
            
        # Remove extra axis 
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()
        
    def plot_correlation(self):
        """Plot correlation matrix for numerical features.
        Consider ordinal and othey types?
        """
        pass

    def _analyze(self):
        df = self.df
        self.n_samples, self.n_features = df.shape

        # update num and cat cols (check if they were user-defined as well)
        num_cols = set(df.select_dtypes(include=["int64", "float64"]).columns.tolist())
        cat_cols = set(df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist())

        num_cols.update(self._feature_overrides["num_cols"])
        cat_cols.update(self._feature_overrides["cat_cols"])

        num_cols.difference_update(self._feature_overrides["cat_cols"])
        cat_cols.difference_update(self._feature_overrides["num_cols"])

        self.num_cols = sorted(list(num_cols))
        self.cat_cols = sorted(list(cat_cols))

        self.cols = self.num_cols + self.cat_cols
        if self.target: self.cat_cols.remove(self.target)
        self.mv_cols = df.columns[df.isna().any()].to_list()
        self.mv_percentage = round(df.isna().sum().sum() / df.size * 100, 2)
        self.duplicate_rows = df.duplicated().sum()

        if self.target and self.target in df.columns:
            self.target_distribution = round(df[self.target].value_counts(normalize=True), 2).to_dict()
        else:
            self.target_distribution = {}

        self.analysis_summary = {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "target": self.target,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "mv_cols": self.mv_cols,
            "mv_percentage": self.mv_percentage,
            "duplicate_rows": self.duplicate_rows,
            "target_distribution": self.target_distribution,
        }

    def _refresh(self):
        self._analyze()

    def reset_feature_type_overrides(self):
        """Clean all overrides to num and cat columns."""
        self._feature_overrides = {"num_cols": set(), "cat_cols": set()}
        self._refresh()

    def _set_semantic_feature_types(self, cont_cols=None, ord_cols=None, nom_cols=None, bin_cols=None):
        """Sets semantic feature types to the profiler.
            If a new setting occurs, it will override the previous.
        """
        
        feature_roles = {
            "continuous": cont_cols,
            "ordinal": ord_cols,
            "nominal": nom_cols,
            "binary": bin_cols
        }

        for role, cols in feature_roles.items():
            if cols is not None:
                setattr(self, role, cols)
                self.metadata_summary[f"{role}_cols"] = cols

    def _override_feature_types(self, numeric=None, categorical=None):
        if numeric:
            self._feature_overrides["num_cols"].update(numeric)
            self._feature_overrides["cat_cols"].difference(numeric)
        if categorical:
            self._feature_overrides["cat_cols"].update(categorical)
            self._feature_overrides["num_cols"].difference_update(categorical)

    def _set_feature_mappings(self, map: dict, inverse_map: dict=None):
        if not hasattr(self, "feature_mappings"):
            self.feature_mappings = {}
        self.feature_mappings.update(map)
        self.metadata_summary["feature_mappings"] = self.feature_mappings
        if inverse_map:
            if not hasattr(self, "inverse_feature_mappings"):
                self.inverse_feature_mappings = {}
            self.inverse_feature_mappings.update(inverse_map)
            self.metadata_summary["inverse_mappings"] = self.inverse_feature_mappings        

    @staticmethod
    def combinatorial_k_2(n_cols: list):
        comb = set()
        for i in n_cols:
            for j in n_cols:
                if i != j and (j, i) not in comb:
                    comb.add((i, j))
        return comb
    
    # TODO: def save_plot(self, plot_func, *args, **kwargs):

    # TODO: get_data()?

# Given this code, how could I use it to plot all features?
# How can I determine feature types? And change them?
# Can I provide alternative maps? map and reverse mapping?

class MissingDataProfiler:
    pass
    # TODO: Get basic stats on missing data and visualization



class DataVisualizer:
    pass





