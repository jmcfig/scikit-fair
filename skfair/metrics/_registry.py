"""Single source of truth for metric metadata.

Every metric the package knows about — both **performance** and **fairness** —
is registered here exactly once, in :data:`REGISTRY`. Any other part of the
package that needs information about a metric (its callable, its family, whether
it is a ratio or a difference, which direction is "better", a human-readable
label, ...) derives that information from this registry rather than keeping its
own copy:

* :mod:`skfair.experimentation` builds its metric registry from here,
* :class:`skfair.audit.FairnessAuditor` and its plots read labels / ideals here,
* :mod:`skfair.comparison` reads metric directions and fairness/performance
  classification here.

Add a metric in *one* place (its function in ``_performance.py`` /
``_fairness.py`` plus an entry below) and every consumer picks it up.
"""

from collections import namedtuple

from . import _fairness as _f
from . import _performance as _p

# func    : the callable implementing the metric
# kind    : "performance" | "fairness"
# family  : "independence" | "separation" | "sufficiency" | "accuracy" for
#           fairness metrics; "performance" for performance metrics
# ideal   : 0 (difference, ideal = 0) or 1 (ratio, ideal = 1) for fairness
#           metrics; "higher" for performance metrics (higher is better)
# display : short human-readable label used by the auditor's tables and plots
# aliases : tuple of accepted alternative names (short forms, literature
#           abbreviations); every alias is also importable from ``skfair.metrics``
MetricSpec = namedtuple(
    "MetricSpec", ["func", "kind", "family", "ideal", "display", "aliases"]
)


def _perf(func, display, aliases=()):
    return MetricSpec(func, "performance", "performance", "higher", display, aliases)


def _fair(func, family, ideal, display, aliases=()):
    return MetricSpec(func, "fairness", family, ideal, display, aliases)


#: The canonical registry: canonical metric name -> :class:`MetricSpec`.
REGISTRY = {
    # ===================== Performance =====================
    "accuracy": _perf(_p.accuracy, "Accuracy"),
    "balanced_accuracy": _perf(_p.balanced_accuracy, "Balanced Accuracy"),
    "geometric_mean": _perf(_p.geometric_mean, "G-Mean"),
    "precision": _perf(_p.precision, "Precision"),
    "recall": _perf(_p.recall, "Recall"),
    "f1_score": _perf(_p.f1_score, "F1"),
    "true_positive_rate": _perf(_p.true_positive_rate, "TPR"),
    "false_positive_rate": _perf(_p.false_positive_rate, "FPR"),
    "true_negative_rate": _perf(_p.true_negative_rate, "TNR"),
    "false_negative_rate": _perf(_p.false_negative_rate, "FNR"),
    "positive_predictive_value": _perf(_p.positive_predictive_value, "PPV"),
    "negative_predictive_value": _perf(_p.negative_predictive_value, "NPV"),
    "false_discovery_rate": _perf(_p.false_discovery_rate, "FDR"),
    "false_omission_rate": _perf(_p.false_omission_rate, "FOR"),
    # ===================== Fairness =====================
    # ----- independence (depends on Ŷ only) -----
    "statistical_parity_difference": _fair(
        _f.statistical_parity_difference, "independence", 0,
        "Statistical Parity Diff", ("spd",),
    ),
    "disparate_impact": _fair(
        _f.disparate_impact, "independence", 1, "Disparate Impact", ("di",),
    ),
    # ----- separation: TPR -----
    "true_positive_rate_difference": _fair(
        _f.true_positive_rate_difference, "separation", 0, "TPR Difference",
        ("eod", "tpr_diff", "equal_opportunity_difference"),
    ),
    "true_positive_rate_ratio": _fair(
        _f.true_positive_rate_ratio, "separation", 1, "TPR Ratio",
        ("eor", "tpr_ratio", "equal_opportunity_ratio"),
    ),
    # ----- separation: FPR -----
    "false_positive_rate_difference": _fair(
        _f.false_positive_rate_difference, "separation", 0, "FPR Difference",
        ("fpr_diff",),
    ),
    "false_positive_rate_ratio": _fair(
        _f.false_positive_rate_ratio, "separation", 1, "FPR Ratio", ("fpr_ratio",),
    ),
    # ----- separation: TNR -----
    "true_negative_rate_difference": _fair(
        _f.true_negative_rate_difference, "separation", 0, "TNR Difference",
        ("tnr_diff",),
    ),
    "true_negative_rate_ratio": _fair(
        _f.true_negative_rate_ratio, "separation", 1, "TNR Ratio", ("tnr_ratio",),
    ),
    # ----- separation: FNR -----
    "false_negative_rate_difference": _fair(
        _f.false_negative_rate_difference, "separation", 0, "FNR Difference",
        ("fnr_diff",),
    ),
    "false_negative_rate_ratio": _fair(
        _f.false_negative_rate_ratio, "separation", 1, "FNR Ratio", ("fnr_ratio",),
    ),
    # ----- separation: combined odds (FPR + TPR) -----
    "average_odds_difference": _fair(
        _f.average_odds_difference, "separation", 0, "Average Odds Diff", ("aod",),
    ),
    "average_odds_ratio": _fair(
        _f.average_odds_ratio, "separation", 1, "Average Odds Ratio", ("aor",),
    ),
    # ----- sufficiency: PPV (predictive parity) -----
    "positive_predictive_value_difference": _fair(
        _f.positive_predictive_value_difference, "sufficiency", 0, "PPV Difference",
        ("ppv_diff",),
    ),
    "positive_predictive_value_ratio": _fair(
        _f.positive_predictive_value_ratio, "sufficiency", 1, "PPV Ratio",
        ("ppv_ratio",),
    ),
    # ----- sufficiency: NPV -----
    "negative_predictive_value_difference": _fair(
        _f.negative_predictive_value_difference, "sufficiency", 0, "NPV Difference",
        ("npv_diff",),
    ),
    "negative_predictive_value_ratio": _fair(
        _f.negative_predictive_value_ratio, "sufficiency", 1, "NPV Ratio",
        ("npv_ratio",),
    ),
    # ----- sufficiency: FDR -----
    "false_discovery_rate_difference": _fair(
        _f.false_discovery_rate_difference, "sufficiency", 0, "FDR Difference",
        ("fdr_diff",),
    ),
    "false_discovery_rate_ratio": _fair(
        _f.false_discovery_rate_ratio, "sufficiency", 1, "FDR Ratio", ("fdr_ratio",),
    ),
    # ----- sufficiency: FOR -----
    "false_omission_rate_difference": _fair(
        _f.false_omission_rate_difference, "sufficiency", 0, "FOR Difference",
        ("for_diff",),
    ),
    "false_omission_rate_ratio": _fair(
        _f.false_omission_rate_ratio, "sufficiency", 1, "FOR Ratio", ("for_ratio",),
    ),
    # ----- accuracy -----
    "accuracy_difference": _fair(
        _f.accuracy_difference, "accuracy", 0, "Accuracy Difference", ("acc_diff",),
    ),
    "accuracy_ratio": _fair(
        _f.accuracy_ratio, "accuracy", 1, "Accuracy Ratio", ("acc_ratio",),
    ),
}


def _build_alias_map():
    """name (canonical or alias) -> canonical name."""
    mapping = {}
    for canonical, spec in REGISTRY.items():
        mapping[canonical] = canonical
        for alias in spec.aliases:
            if alias in mapping:
                raise ValueError(f"Duplicate metric name/alias: {alias!r}")
            mapping[alias] = canonical
    return mapping


#: Every accepted name (canonical names and aliases) -> canonical name.
_CANONICAL = _build_alias_map()


# ---------------------------------------------------------------------------
# Accessors — the public way to ask the registry about a metric
# ---------------------------------------------------------------------------

def all_names():
    """Return every accepted metric name (canonical names and aliases)."""
    return list(_CANONICAL)


def resolve(name):
    """Return the canonical name for *name* (a canonical name or alias)."""
    try:
        return _CANONICAL[name]
    except KeyError:
        raise KeyError(
            f"Unknown metric '{name}'. Use a canonical name from REGISTRY "
            f"or a documented alias."
        ) from None


def spec(name):
    """Return the :class:`MetricSpec` for *name* (canonical or alias)."""
    return REGISTRY[resolve(name)]


def get_func(name):
    """Resolve a metric name (canonical or alias) to its callable."""
    return REGISTRY[resolve(name)].func


def kind_of(name, default="performance"):
    """Return ``"fairness"`` or ``"performance"`` for *name*.

    Unknown names (e.g. result columns from outside skfair) return *default*.
    """
    canonical = _CANONICAL.get(name)
    return REGISTRY[canonical].kind if canonical is not None else default


def is_fairness(name):
    """Return ``True`` if *name* is a known fairness metric."""
    canonical = _CANONICAL.get(name)
    return canonical is not None and REGISTRY[canonical].kind == "fairness"


def direction(name, default=None):
    """Return the "better" direction for *name*.

    ``"higher"`` (higher is better, performance metrics), ``"zero"`` (closer to
    0 is better, difference metrics) or ``"one"`` (closer to 1 is better, ratio
    metrics). Unknown names return *default*.
    """
    canonical = _CANONICAL.get(name)
    if canonical is None:
        return default
    ideal = REGISTRY[canonical].ideal
    if ideal == "higher":
        return "higher"
    return "zero" if ideal == 0 else "one"


# ---------------------------------------------------------------------------
# Fairness-only views + the benchmark helper
# ---------------------------------------------------------------------------

#: Fairness-only view of :data:`REGISTRY` (canonical name -> spec). Kept as a
#: public name for backward compatibility; specs expose ``.func`` / ``.family``
#: / ``.ideal``.
METRICS = {name: s for name, s in REGISTRY.items() if s.kind == "fairness"}

#: Default subset evaluated by :func:`benchmark` when no ``metrics`` are given.
#: One representative *difference* per family, with separation keeping both of its
#: equalized-odds components (TPR and FPR). The full grid is available via
#: ``metrics="all"`` or by naming metrics explicitly.
DEFAULT_BENCHMARK_METRICS = [
    "statistical_parity_difference",         # independence
    "equal_opportunity_difference",          # separation (TPR)
    "false_positive_rate_difference",        # separation (FPR)
    "positive_predictive_value_difference",  # sufficiency
]


def benchmark(y_true, y_pred, sensitive_attr, metrics=None):
    """Evaluate a selection of group-fairness metrics in one call.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_attr : np.ndarray
        Binary group indicator (1 = privileged, 0 = unprivileged).
    metrics : None, "all", or iterable of str, default=None
        Which metrics to compute.

        * ``None`` — the sensible default subset
          (:data:`DEFAULT_BENCHMARK_METRICS`), one representative per family.
        * ``"all"`` — the complete grid of every fairness metric in
          :data:`METRICS`.
        * an iterable of names — those metrics, given by canonical name or any
          documented alias.

    Returns
    -------
    dict of str -> float
        Mapping of the requested metric names to their values.
    """
    if metrics is None:
        names = list(DEFAULT_BENCHMARK_METRICS)
    elif metrics == "all":
        names = list(METRICS)
    else:
        names = list(metrics)

    return {
        name: get_func(name)(y_true, y_pred, sensitive_attr)
        for name in names
    }
