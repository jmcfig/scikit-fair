"""
Registries for datasets, methods, and metrics used by the Experiment class.

Each registry maps human-readable names to dotted import paths and metadata,
resolved lazily at runtime via ``_import_object()``.
"""

import importlib


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
        "sens_attr": "sex",
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
    "GeometricFairnessRepair": {
        "path": "skfair.preprocessing.GeometricFairnessRepair",
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
METRIC_REGISTRY = {
    "accuracy": {
        "path": "skfair.metrics.accuracy",
        "type": "performance",
    },
    "balanced_accuracy": {
        "path": "skfair.metrics.balanced_accuracy",
        "type": "performance",
    },
    "disparate_impact": {
        "path": "skfair.metrics.disparate_impact",
        "type": "fairness",
    },
    "spd": {
        "path": "skfair.metrics.statistical_parity_difference",
        "type": "fairness",
    },
    "eod": {
        "path": "skfair.metrics.equal_opportunity_difference",
        "type": "fairness",
    },
    "aod": {
        "path": "skfair.metrics.average_odds_difference",
        "type": "fairness",
    },
}


def _import_object(dotted_path):
    """Import and return the object at *dotted_path*.

    Example: ``_import_object("sklearn.svm.SVC")`` → ``<class SVC>``.
    """
    module_path, _, attr_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)
