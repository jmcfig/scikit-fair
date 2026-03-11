"""
Experimentation module — automates fairness-method comparison experiments.
"""

from ._experiment import Experiment
from ._registry import DATASET_REGISTRY, METHOD_REGISTRY, METRIC_REGISTRY

__all__ = [
    "Experiment",
    "DATASET_REGISTRY",
    "METHOD_REGISTRY",
    "METRIC_REGISTRY",
]
