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

Scatter plot of fairness distance from ideal vs. performance metric for each method, faceted by dataset. The x-axis is `|metric|` for difference metrics (ideal = 0, e.g. `spd`, `eod`) and `|metric - 1|` for ratio metrics (ideal = 1, e.g. `disparate_impact`, `equal_opportunity_ratio`), so **lower x is always fairer** and the top-left corner is always ideal. Helps identify methods that achieve a good balance.

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

Generates a self-contained interactive HTML file with embedded matplotlib charts (base64-encoded), tab navigation, filtering checkboxes, and a PDF export button.

### Report tabs

The HTML report is organized into five tabs:

| Tab | Contents |
|---|---|
| **Tables** | Summary tables for every metric, with methods as rows and datasets as columns |
| **Performance** | Bar charts for each performance metric, grouped by dataset and method |
| **Fairness** | Bar charts for each fairness metric, with automatic reference lines at zero |
| **Rankings** | Heatmap of method rankings across datasets and metrics (lower rank = better) |
| **Tradeoff** | Scatter plot of fairness vs. performance, faceted by dataset |

### Interactive controls

Each tab includes **filtering checkboxes** for datasets, classifiers, and metrics. Use the **Select All / Deselect All** buttons to quickly toggle visibility. Filtering is applied client-side — no re-generation needed.

### PDF export

Click the **Export PDF** button in the top-right corner to produce a print-friendly version. Collapsed sections stay collapsed in the PDF output.

### `to_html()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | — | Output file path (e.g. `"report.html"`) |
| `datasets` | `list[str]` | `None` | Filter to specific datasets |
| `methods` | `list[str]` | `None` | Filter to specific methods |
| `classifiers` | `list[str]` | `None` | Filter to specific classifiers |
| `metrics` | `list[str]` | `None` | Metrics to include (defaults to all detected) |
| `fairness_metric` | `str` | `"spd"` | Fairness metric for tradeoff and detail charts |
| `performance_metric` | `str` | `"accuracy"` | Performance metric for tradeoff chart |
| `classifier` | `str` | `None` | Classifier aggregation: `None`/`"average"`, `"best"`, or a specific name |

```python
# Full example with all parameters
report.to_html(
    "report.html",
    datasets=["adult"],
    methods=["Massaging", "FairSmote"],
    fairness_metric="spd",
    performance_metric="accuracy",
    classifier="best",
)
```

!!! note "Future benchmark"
    The comparison module will be expanded as the authors design their own comprehensive fairness preprocessing benchmark. Future versions will include additional analysis tools and visualization options.
