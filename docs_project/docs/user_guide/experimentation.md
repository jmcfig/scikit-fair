# Experimentation

The `skfair.experimentation` module provides the `Experiment` class for running automated dataset × method × classifier comparison experiments with cross-validation.

---

## Python API

### Basic usage

```python
from skfair.experimentation import Experiment

exp = Experiment(
    datasets=["adult", "compas"],
    methods=["Massaging", "FairSmote", "ReweighingClassifier"],
    n_splits=5,
)
results = exp.run(verbose=True)
print(results)
```

`run()` returns a DataFrame with one row per (dataset, method, classifier) combination and `{metric}_mean` / `{metric}_std` columns.

### Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `datasets` | `["adult"]` | List of dataset names from `DATASET_REGISTRY` |
| `methods` | All registered | List of method names from `METHOD_REGISTRY` |
| `classifiers` | LogisticRegression | Dict `{"name": estimator}` or list of dotted paths |
| `metrics` | All registered | List of metric keys from `METRIC_REGISTRY` |
| `n_splits` | `5` | Number of CV folds (1 = single train/test split) |
| `random_state` | `42` | Random seed |
| `dataset_config` | `None` | Per-dataset overrides, e.g., `{"adult": {"sens_attr": "race"}}` |
| `method_config` | `None` | Per-method parameter overrides |
| `audit_bias` | `False` | Create a `BiasAuditor` per dataset |
| `audit_fairness` | `False` | Store out-of-fold predictions for `FairnessAuditor` |
| `save_results` | `False` | Write results CSV after `run()` |
| `save_object` | `False` | Pickle full `Experiment` after `run()` |
| `save_path` | `"experiment"` | Base path for saved files |

---

## XML configuration

Experiments can also be defined via XML configuration files:

```python
exp = Experiment.from_xml("config.xml")
results = exp.run()
```

Or pass the XML path directly to the constructor:

```python
exp = Experiment(xml="config.xml")
```

---

## Post-run analysis

### ComparisonReport

Convert results to a visual comparison report (see [Comparison](comparison.md)):

```python
report = exp.to_report()
report.plot_performance()
report.plot_tradeoff(fairness_metric="spd", performance_metric="accuracy")
```

### FairnessAuditor

Get a `FairnessAuditor` for a specific (dataset, method, classifier) combination (requires `audit_fairness=True`):

```python
exp = Experiment(
    datasets=["adult"],
    methods=["Massaging"],
    audit_fairness=True,
)
exp.run()

fa = exp.get_fairness_auditor("adult", "Massaging", "LogisticRegression")
print(fa.fairness_metrics())
fa.plot_fairness_radar()
```

---

## Save and load

### Saving

```python
# Via constructor flags
exp = Experiment(
    datasets=["adult"],
    save_results=True,   # saves {save_path}.csv
    save_object=True,    # saves {save_path}.pkl
    save_path="my_experiment",
)
exp.run()

# Or manually after run
exp.save(path="my_experiment", results=True, object=True)
```

### Loading

```python
exp = Experiment.load("my_experiment.pkl")
print(exp.results_)
```

---

## Registries

Three registries define what is available by name in experiments:

### DATASET_REGISTRY

```python
from skfair.experimentation import DATASET_REGISTRY
print(list(DATASET_REGISTRY.keys()))
# ['adult', 'compas', 'german', 'heart_disease', 'ricci']
```

Each entry specifies the loader function, default `sens_attr`, and `priv_group`.

### METHOD_REGISTRY

```python
from skfair.experimentation import METHOD_REGISTRY
print(list(METHOD_REGISTRY.keys()))
# ['Baseline', 'Massaging', 'FairSmote', 'FairOversampling', 'FAWOS',
#  'HeterogeneousFOS', 'FairwayRemover', 'GeometricFairnessRepair',
#  'LearningFairRepresentations', 'ReweighingClassifier',
#  'FairBalanceClassifier', 'FairMask']
```

### METRIC_REGISTRY

```python
from skfair.experimentation import METRIC_REGISTRY
print(list(METRIC_REGISTRY.keys()))
# ['accuracy', 'balanced_accuracy', 'disparate_impact', 'spd', 'eod', 'aod']
```

Each entry maps a short key to a metric function and its type (performance or fairness).
