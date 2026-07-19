# Changelog

## 0.2.0 (upcoming)

### Added

- **ACSIncome dataset** — new `skfair.datasets.fetch_acs_income` loader for
  the large-scale Adult successor of Ding et al. (2021): the 2018 1-year ACS
  PUMS release (1,664,500 rows, all US states and Puerto Rico), with the
  binary income > $50K target, `SEX` encoded 1 = male, and the multi-valued
  `RAC1P` race column preserved for pairwise group comparisons. Unlike the
  bundled datasets, it is fetched once from OpenML and cached locally
  (scikit-learn `fetch_*` convention); an optional `subsample` /
  `random_state` pair returns a tractable subset. Registered in
  `DATASET_REGISTRY` as `"acs_income"`.
- **Consistent handling of multi-valued sensitive attributes in
  experiments** — `run_cv` (and therefore `Experiment`) now binarises the
  sensitive attribute privileged-vs-rest via the per-dataset `priv_group`
  before computing fairness metrics and storing out-of-fold predictions,
  mirroring the auditors' behaviour. No-op for already-binary 0/1 columns
  with the default `priv_group=1`.
- **Metric input validation** — fairness metric functions now raise a
  `ValueError` when the group indicator contains values other than 0/1
  (previously such samples were silently ignored), with guidance to
  binarise via `(sens == priv_group)` or `IntersectionalBinarizer`.
- **Pairwise group selection in fairness metrics** — every fairness metric
  (and the `benchmark` helper) accepts optional `priv_group` /
  `unpriv_group` arguments for multi-valued sensitive attributes:
  `priv_group` alone compares privileged-vs-rest; together they compare
  exactly that pair of groups, excluding the others, so any pairwise
  combination can be inspected. `FairnessAuditor` gains the matching
  `unpriv_group` parameter.
- **Per-dataset `priv_group` reaches the methods** — `build_pipeline` (and
  therefore `Experiment`) auto-injects the dataset's `priv_group` into
  methods whose constructor accepts it, overriding the registry default
  but never an explicit `method_config` entry.

### Changed

- **`DisparateImpactRemover.fit` no longer quadratic** — the quantile-edge
  computation (`np.quantile` with ~N_min points triggered numpy's multi-kth
  partition) and the per-bucket median scans are replaced by a single sort
  per (column, group) with direct interpolation/segment reads. Outputs are
  bit-identical; fitting 100k rows drops from minutes to under half a second.
- **`FairBalance` weight assignment vectorised** — the per-row lookup loop
  is replaced by (group, label) cell masks; identical weights, ~50× faster
  on large data.
- **`Reweighing` weight assignment vectorised** — same per-row loop pattern
  replaced by (group, label) cell masks (found by the full-catalogue runtime
  screening); identical weights, ~80× faster on large data.
  `ReweighingClassifier` benefits automatically.
- **`FairOversampling` generation batched** — synthetic samples are now
  produced with a single KNN query and matrix interpolation instead of a
  per-sample Python loop (found by comparing against the authors' reference
  code, which is batched). The per-sample procedure is unchanged (uniform
  non-self neighbour pick, one interpolation factor per sample, identical
  cell balancing), ~16× faster on large data. Note: the RNG draw order
  changes, so seeded outputs differ sample-wise from previous versions.
- **`Reweighing` / `ReweighingClassifier` no longer take `priv_group`**
  (breaking) — the parameter was stored but unused: Kamiran's weighting
  formula corrects every group value towards statistical independence and
  needs no designated privileged group.

- **Counterpart fairness metrics** — every base measure now has both a
  difference form (suffix `_difference`, ideal = 0) and a ratio form
  (suffix `_ratio`, ideal = 1), across four families: independence,
  separation, **sufficiency** (PPV/NPV/FDR/FOR — new), and accuracy.
  Canonical names always use a `_difference` / `_ratio` suffix (never
  `_parity`); `equal_opportunity_difference` / `_ratio` remain as TPR
  aliases. New performance metrics `geometric_mean`,
  `positive_predictive_value`, `negative_predictive_value`,
  `false_discovery_rate`, and `false_omission_rate`.
- **Single metric registry** — all metric metadata (function, family,
  ideal value, display label, aliases) now lives in one place,
  `skfair.metrics.REGISTRY`, with a `benchmark` helper and `METRICS`
  view. The experimentation `METRIC_REGISTRY`, `FairnessAuditor`, and
  `skfair.comparison` all derive their metric information from this
  registry instead of keeping separate copies.
- `FairnessAuditor.fairness_metrics` now covers the full registry,
  including the new sufficiency family.

### Fixed

- `skfair.metrics` failed to import because `predictive_parity_difference`
  / `predictive_parity_ratio` were exported but never defined. These
  predictive-parity aliases have been removed; use
  `positive_predictive_value_difference` / `_ratio` (aliases `ppv_diff`
  / `ppv_ratio`).
- `ComparisonReport.plot_tradeoff` now plots ratio fairness metrics as
  `|metric - 1|` instead of `|metric|`, so "lower x = fairer" holds for
  every metric.
- Ratio metrics are classified with the `"one"` direction in
  `DEFAULT_METRIC_DIRECTION` (derived from the registry), correcting
  their ranking and "best classifier" aggregation in `plot_ranking`
  and `summary_tables(classifier="best")`.

### Documentation

- Docstrings of `plot_tradeoff`, `_plot_tradeoff_scatter`, and
  `plot_ranking` (`higher_is_better` parameter) updated to describe
  the direction-aware behavior and the `"higher"`/`"zero"`/`"one"`
  taxonomy.
- README and `docs_project/docs/user_guide/comparison.md` tradeoff
  sections updated. README's "top-right corner" typo corrected to
  "top-left corner".

## 0.1.2

### Fixed

- **Dataset loaders now ship their data.** `load_adult`, `load_compas`,
  `load_german`, `load_heart_disease`, and `load_ricci` previously raised
  `FileNotFoundError` when installed from PyPI because the CSV/`.dat` files
  under `skfair/datasets/data` were never bundled. Added `MANIFEST.in` and
  `[tool.setuptools.package-data]` so the data ships in both the wheel and
  the sdist.
- Backported the ratio-based tradeoff-plot fix so
  `ComparisonReport.plot_tradeoff` plots ratio metrics as `|metric - 1|`.

## 0.1.0

### Added

- **Auditing Tools** — Class for dataset and model audits, including bias detection and fairness checks.
- **Comparison Module** — `Comparison` class for side-by-side metric comparison across models and datasets.
- **Experimentation framework** — `Experiment` class with `run_cv` for cross-validated fairness evaluation, YAML/dict config parsing, and structured results.
- **Save & export** — CSV results export, pickle model saving, and HTML report generation.
- **Clean Example notebooks** — 8 Jupyter notebooks demonstrating core workflows.

## 0.0.1

Initial development releases with base package structure.
