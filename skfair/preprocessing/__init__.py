"""Fairness-aware preprocessing algorithms."""

from ._base import BaseFairSampler
from ._drop_columns import DropColumns
from ._intersectional import IntersectionalBinarizer
from ._massaging import Massaging
from ._reweighing import Reweighing
from ._reweighing_classifier import ReweighingClassifier
from ._fairbalance import FairBalance
from ._fairbalance_classifier import FairBalanceClassifier
from ._disparate_impact_remover import GeometricFairnessRepair
from ._fair_smote import FairSmote
from ._fairway import FairwayRemover
from ._fos import FairOversampling
from ._fawos import FAWOS
from ._heterogeneous_fos import HeterogeneousFOS
from ._fairmask import FairMask
from ._optimized_preprocessing import OptimizedPreprocessing
from ._learning_fair_representations import LearningFairRepresentations

__all__ = [
    "BaseFairSampler",
    "DropColumns",
    "IntersectionalBinarizer",
    "Massaging",
    "Reweighing",
    "ReweighingClassifier",
    "FairBalance",
    "FairBalanceClassifier",
    "GeometricFairnessRepair",
    "FairSmote",
    "FairwayRemover",
    "FairOversampling",
    "FAWOS",
    "HeterogeneousFOS",
    "FairMask",
    "OptimizedPreprocessing",
    "LearningFairRepresentations",
]