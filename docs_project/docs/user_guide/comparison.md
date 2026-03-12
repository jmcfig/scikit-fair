# Comparison

The `skfair.comparison` module provides `ComparisonReport`, a visualisation tool for comparing multiple fairness preprocessing methods across datasets and classifiers.

---

## Creating a report

`ComparisonReport` takes a results DataFrame with columns `dataset`, `method`, `classifier`, plus `{metric}_mean` and `{metric}_std` column pairs.

The easiest way to get this DataFrame is from an `Experiment` (see [Experimentation](experimentation.md)):

```python
from skfair.experimentation import Experiment
from skfair.comparison import ComparisonReport

exp = Experiment(
    datasets=["adult", "compas"],
    methods=["Massaging", "FairSmote", "ReweighingClassifier"],
    n_splits=5,
)
results = exp.run()

report = ComparisonReport(results)
# or equivalently:
report = exp.to_report()
```

On construction, `ComparisonReport` auto-detects which columns are performance metrics and which are fairness metrics.

---

## Summary tables

```python
tables = report.summary_tables()
```

Returns a dictionary of DataFrames — one per metric — with method means averaged over classifiers, pivoted by dataset.

---

## Plot methods

All plot methods return `(fig, axes)` tuples.

### Performance comparison

```python
report.plot_performance()
```

Grouped bar charts showing performance metrics (e.g., accuracy, balanced accuracy) for each method, faceted by dataset.

### Fairness averaged

```python
report.plot_fairness_averaged(metric="spd")
```

Bar chart of a single fairness metric averaged over classifiers, for each method per dataset.

### Fairness detailed

```python
report.plot_fairness_detailed(metric="spd")
```

Grouped bars showing a fairness metric per classifier, for each method per dataset.

### Fairness–performance tradeoff

```python
report.plot_tradeoff(fairness_metric="spd", performance_metric="accuracy")
```

Scatter plot of |fairness metric| vs. performance metric for each method, faceted by dataset. Helps identify methods that achieve a good balance.

### Method ranking

```python
report.plot_ranking()
```

Heatmap of method rankings per dataset across all metrics. Lower rank (closer to 1) is better.

### All plots at once

```python
report.plot_all(fairness_metric="spd")
```

Runs all five plot methods and returns a list of `(fig, axes)` tuples.
