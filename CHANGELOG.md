# Changelog

## Unreleased

### Added

- **Counterpart fairness metrics** — every base measure now has both a
  difference form (ideal = 0) and a ratio/parity form (ideal = 1):
  `false_positive_rate_difference`, `true_negative_rate_parity`,
  `false_negative_rate_parity`, `accuracy_difference`, and
  `average_odds_ratio`. `false_positive_rate_parity` is also exported
  as an alias of `predictive_equality`. `skfair.metrics`,
  `FairnessAuditor.fairness_metrics`, `ComparisonReport`, and the
  experimentation `METRIC_REGISTRY` all expose the new entries; the
  modules are reorganised so counterparts appear side by side.

### Fixed

- `ComparisonReport.plot_tradeoff` now plots ratio fairness metrics
  (`disparate_impact`, `equal_opportunity_ratio`, `predictive_equality`,
  `accuracy_parity`) as `|metric - 1|` instead of `|metric|`, so
  "lower x = fairer" holds for every metric.
- `predictive_equality` and `accuracy_parity` are now classified as
  ratio (`"one"` direction) in `DEFAULT_METRIC_DIRECTION`, matching
  their implementations in `skfair.metrics`. This corrects their
  ranking and "best classifier" aggregation in `plot_ranking` and
  `summary_tables(classifier="best")`.

### Documentation

- Docstrings of `plot_tradeoff`, `_plot_tradeoff_scatter`, and
  `plot_ranking` (`higher_is_better` parameter) updated to describe
  the direction-aware behavior and the `"higher"`/`"zero"`/`"one"`
  taxonomy.
- README and `docs_project/docs/user_guide/comparison.md` tradeoff
  sections updated. README's "top-right corner" typo corrected to
  "top-left corner".

## 0.1.0

### Added

- **Auditing Tools** — Class for dataset and model audits, including bias detection and fairness checks.
- **Comparison Module** — `Comparison` class for side-by-side metric comparison across models and datasets.
- **Experimentation framework** — `Experiment` class with `run_cv` for cross-validated fairness evaluation, YAML/dict config parsing, and structured results.
- **Save & export** — CSV results export, pickle model saving, and HTML report generation.
- **Clean Example notebooks** — 8 Jupyter notebooks demonstrating core workflows.

## 0.0.1

Initial development releases with base package structure.
