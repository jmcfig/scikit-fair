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
METRIC_REGISTRY = {
    # --- Performance ---
    "accuracy": {
        "path": "skfair.metrics.accuracy",
        "type": "performance",
    },
    "balanced_accuracy": {
        "path": "skfair.metrics.balanced_accuracy",
        "type": "performance",
    },
    "precision": {
        "path": "skfair.metrics.precision",
        "type": "performance",
    },
    "recall": {
        "path": "skfair.metrics.recall",
        "type": "performance",
    },
    "f1_score": {
        "path": "skfair.metrics.f1_score",
        "type": "performance",
    },
    "true_positive_rate": {
        "path": "skfair.metrics.true_positive_rate",
        "type": "performance",
    },
    "false_positive_rate": {
        "path": "skfair.metrics.false_positive_rate",
        "type": "performance",
    },
    "true_negative_rate": {
        "path": "skfair.metrics.true_negative_rate",
        "type": "performance",
    },
    "false_negative_rate": {
        "path": "skfair.metrics.false_negative_rate",
        "type": "performance",
    },
    # --- Fairness ---
    # Organised in counterpart pairs: difference (ideal = 0) and
    # ratio/parity (ideal = 1) form for each base measure.
    # Positive prediction rate
    "spd": {
        "path": "skfair.metrics.statistical_parity_difference",
        "type": "fairness",
    },
    "disparate_impact": {
        "path": "skfair.metrics.disparate_impact",
        "type": "fairness",
    },
    # TPR
    "eod": {
        "path": "skfair.metrics.equal_opportunity_difference",
        "type": "fairness",
    },
    "equal_opportunity_ratio": {
        "path": "skfair.metrics.equal_opportunity_ratio",
        "type": "fairness",
    },
    # FPR
    "false_positive_rate_difference": {
        "path": "skfair.metrics.false_positive_rate_difference",
        "type": "fairness",
    },
    "predictive_equality": {
        "path": "skfair.metrics.predictive_equality",
        "type": "fairness",
    },
    # TNR
    "true_negative_rate_difference": {
        "path": "skfair.metrics.true_negative_rate_difference",
        "type": "fairness",
    },
    "true_negative_rate_parity": {
        "path": "skfair.metrics.true_negative_rate_parity",
        "type": "fairness",
    },
    # FNR
    "false_negative_rate_difference": {
        "path": "skfair.metrics.false_negative_rate_difference",
        "type": "fairness",
    },
    "false_negative_rate_parity": {
        "path": "skfair.metrics.false_negative_rate_parity",
        "type": "fairness",
    },
    # Accuracy
    "accuracy_difference": {
        "path": "skfair.metrics.accuracy_difference",
        "type": "fairness",
    },
    "accuracy_parity": {
        "path": "skfair.metrics.accuracy_parity",
        "type": "fairness",
    },
    # Combined (FPR + TPR)
    "aod": {
        "path": "skfair.metrics.average_odds_difference",
        "type": "fairness",
    },
    "average_odds_ratio": {
        "path": "skfair.metrics.average_odds_ratio",
        "type": "fairness",
    },
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
