"""
Registries for datasets, methods, and metrics used by the Experiment class.

Each registry maps human-readable names to dotted import paths and metadata,
resolved lazily at runtime via ``_import_object()``.
"""

import importlib

from skfair.metrics import _registry as _metrics


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "adult": {
        "loader": "skfair.datasets.load_adult",
        "sens_attr": "sex",
        "priv_group": 1,
    },
    "compas": {
        "loader": "skfair.datasets.load_compas",
        "sens_attr": "race",
        "priv_group": 1,
    },
    "german": {
        "loader": "skfair.datasets.load_german",
        "sens_attr": "sex",
        "priv_group": 1,
    },
    "heart_disease": {
        "loader": "skfair.datasets.load_heart_disease",
        "sens_attr": "sex",
        "priv_group": 1,
    },
    "ricci": {
        "loader": "skfair.datasets.load_ricci",
        "sens_attr": "Race",
        "priv_group": 1,
    },
}

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
METHOD_REGISTRY = {
    "Baseline": {
        "path": None,
        "category": "baseline",
        "defaults": {},
    },
    "Massaging": {
        "path": "skfair.preprocessing.Massaging",
        "category": "sampler",
        "defaults": {"priv_group": 1},
    },
    "FairSmote": {
        "path": "skfair.preprocessing.FairSmote",
        "category": "sampler",
        "defaults": {"random_state": 42},
    },
    "FairOversampling": {
        "path": "skfair.preprocessing.FairOversampling",
        "category": "sampler",
        "defaults": {"priv_group": 1, "random_state": 42},
    },
    "FAWOS": {
        "path": "skfair.preprocessing.FAWOS",
        "category": "sampler",
        "defaults": {"priv_group": 1, "random_state": 42},
    },
    "HeterogeneousFOS": {
        "path": "skfair.preprocessing.HeterogeneousFOS",
        "category": "sampler",
        "defaults": {"random_state": 42},
    },
    "FairwayRemover": {
        "path": "skfair.preprocessing.FairwayRemover",
        "category": "sampler",
        "defaults": {"priv_group": 1},
    },
    "DisparateImpactRemover": {
        "path": "skfair.preprocessing.DisparateImpactRemover",
        "category": "repair",
        "defaults": {"lambda_param": 1.0},
    },
    "LearningFairRepresentations": {
        "path": "skfair.preprocessing.LearningFairRepresentations",
        "category": "repair",
        "defaults": {"priv_group": 1, "random_state": 42},
    },
    "ReweighingClassifier": {
        "path": "skfair.preprocessing.ReweighingClassifier",
        "category": "meta",
        "defaults": {},
    },
    "FairBalanceClassifier": {
        "path": "skfair.preprocessing.FairBalanceClassifier",
        "category": "meta",
        "defaults": {},
    },
    "FairMask": {
        "path": "skfair.preprocessing.FairMask",
        "category": "meta",
        "defaults": {"random_state": 42},
    },
}

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------
# Derived from the single source of truth in ``skfair.metrics._registry`` so
# there is only one place that knows about metrics. Every canonical name and
# alias is accepted; ``path`` resolves the callable from ``skfair.metrics`` and
# ``type`` is "performance" or "fairness".
METRIC_REGISTRY = {
    name: {"path": f"skfair.metrics.{name}", "type": _metrics.kind_of(name)}
    for name in _metrics.all_names()
}

# Metrics used when the caller does not specify a list. Kept stable at the
# original 6 so the public default output shape does not change when new
# entries are added to METRIC_REGISTRY.
DEFAULT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "disparate_impact",
    "spd",
    "eod",
    "aod",
]


def _import_object(dotted_path):
    """Import and return the object at *dotted_path*.

    Example: ``_import_object("sklearn.svm.SVC")`` → ``<class SVC>``.
    """
    module_path, _, attr_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)
