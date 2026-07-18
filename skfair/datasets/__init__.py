"""Dataset loaders for common fairness benchmarks."""

from ._acs_income import fetch_acs_income
from ._adult import load_adult
from ._compas import load_compas
from ._german import load_german
from ._heart_disease import load_heart_disease
from ._ricci import load_ricci

__all__ = [
    "fetch_acs_income",
    "load_adult",
    "load_compas",
    "load_german",
    "load_heart_disease",
    "load_ricci",
]
