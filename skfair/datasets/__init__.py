"""Dataset loaders for common fairness benchmarks."""

from ._adult import load_adult
from ._german import load_german
from ._heart_disease import load_heart_disease

__all__ = [
    "load_adult",
    "load_german",
    "load_heart_disease",
]
