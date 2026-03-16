# Comparison

The `skfair.comparison` module provides `ComparisonReport`, a visualisation tool for comparing multiple fairness preprocessing methods across datasets and classifiers.

---

## Creating a report

`ComparisonReport` takes a results DataFrame with columns `dataset`, `method`, `classifier`, plus one column per metric (e.g. `accuracy`, `spd`). Optional `{metric}_std` columns are preserved but not used in plots.

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

You can also filter the report at construction time:

```python
report = ComparisonReport(
    results,
    datasets=["adult"],
    methods=["Massaging", "FairSmote"],
    classifiers=["LogReg"],
)
```

---

## Summary tables

```python
tables = report.summary_tables()
```

Returns a dictionary of DataFrames — one per metric — with method means averaged over classifiers, pivoted by dataset.

The `classifier` parameter controls how classifiers are aggregated:

- `None` or `"average"` (default) — average over all classifiers.
- `"best"` — keep only the best-performing classifier per method (by accuracy).
- A specific name (e.g. `"LogReg"`) — filter to that classifier only.

```python
# Only show the best classifier per method
tables = report.summary_tables(classifier="best")

# Filter to a specific classifier
tables = report.summary_tables(classifier="LogReg")
```

---

## Plot methods

All plot methods return `(fig, axes)` tuples.

### Metric bar chart

```python
report.plot_metric_bar(metric="accuracy")   # performance
report.plot_metric_bar(metric="spd")        # fairness (auto reference line)
```

Grouped bar chart for any single metric across datasets. For fairness metrics, a reference line is added automatically.

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

The `classifier` parameter works the same way as in `summary_tables()`:

```python
# Rank using the best classifier per method
report.plot_ranking(classifier="best")

# Rank using a specific classifier
report.plot_ranking(classifier="LogReg")
```

### All plots at once

```python
report.plot_all(fairness_metric="spd")
```

Runs all plot methods (one per metric, plus tradeoff and ranking) and returns a list of `(fig, axes)` tuples.

---

## HTML report

```python
report.to_html("report.html")
```

Generates a self-contained interactive HTML file with embedded charts, tab navigation, filtering checkboxes, and a PDF export button. Collapsed sections stay collapsed in PDF.

You can pass the same filtering parameters as the other methods:

```python
report.to_html("report.html", datasets=["adult"], fairness_metric="spd")
```
