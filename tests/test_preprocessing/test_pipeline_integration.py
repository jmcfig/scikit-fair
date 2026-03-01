"""
Pipeline integration tests for scikit-fair preprocessing classes.

Tests that all classes work correctly in sklearn/imblearn pipelines.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV

from skfair.preprocessing import (
    GeometricFairnessRepair,
    IntersectionalBinarizer,
    ReweighingClassifier,
    FairSmote,
    FairOversampling,
    FAWOS,
    HeterogeneousFOS,
    Massaging,
    FairwayRemover,
    FairMask,
    OptimizedPreprocessing,
    LearningFairRepresentations,
)


class TestSklearnPipelineTransformers:
    """Tests for transformers in sklearn.pipeline.Pipeline."""

    def test_geometric_repair_in_pipeline(self, larger_binary_data, sens_attr):
        """GeometricFairnessRepair works in sklearn Pipeline."""
        X, y = larger_binary_data

        pipe = SklearnPipeline([
            ('repair', GeometricFairnessRepair(
                sens_attr=sens_attr,
                repair_columns=['age', 'income'],
                lambda_param=1.0
            )),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_intersectional_binarizer_in_pipeline(self, larger_binary_data):
        """IntersectionalBinarizer works in sklearn Pipeline."""
        X, y = larger_binary_data

        pipe = SklearnPipeline([
            ('binarizer', IntersectionalBinarizer(
                privileged_definition={'group': 1},
                group_col_name='is_privileged'
            )),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)

    def test_reweighing_classifier_in_pipeline(self, larger_binary_data, sens_attr):
        """ReweighingClassifier works as final estimator in sklearn Pipeline."""
        X, y = larger_binary_data

        # ReweighingClassifier is a meta-estimator, can be final step
        pipe = SklearnPipeline([
            ('clf', ReweighingClassifier(
                estimator=LogisticRegression(solver='liblinear', max_iter=200),
                sens_attr=sens_attr
            ))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)
        proba = pipe.predict_proba(X)

        assert len(preds) == len(X)
        assert proba.shape == (len(X), 2)


    def test_lfr_in_pipeline(self, larger_binary_data, sens_attr):
        """LearningFairRepresentations works in sklearn Pipeline."""
        X, y = larger_binary_data

        pipe = SklearnPipeline([
            ('lfr', LearningFairRepresentations(
                sens_attr=sens_attr,
                priv_group=1,
                k=3,
                maxiter=300,
                maxfun=300,
                random_state=42,
            )),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_lfr_then_classifier_cross_val(self, larger_binary_data, sens_attr):
        """LFR + classifier pipeline works with cross-validation."""
        X, y = larger_binary_data

        pipe = SklearnPipeline([
            ('lfr', LearningFairRepresentations(
                sens_attr=sens_attr,
                priv_group=1,
                k=3,
                maxiter=300,
                maxfun=300,
                random_state=42,
            )),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        scores = cross_val_score(pipe, X, y, cv=3, scoring='accuracy')
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)


class TestImbPipelineResamplers:
    """Tests for resamplers in imblearn.pipeline.Pipeline."""

    def test_fair_smote_in_pipeline(self, larger_binary_data, sens_attr):
        """FairSmote works in imblearn Pipeline."""
        X, y = larger_binary_data

        pipe = ImbPipeline([
            ('sampler', FairSmote(sens_attr=sens_attr, random_state=42)),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_fair_oversampling_in_pipeline(self, larger_binary_data, sens_attr, priv_group):
        """FairOversampling works in imblearn Pipeline."""
        X, y = larger_binary_data

        pipe = ImbPipeline([
            ('sampler', FairOversampling(
                sens_attr=sens_attr,
                priv_group=priv_group,
                random_state=42
            )),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)

    def test_fawos_in_pipeline(self, larger_binary_data, sens_attr, priv_group):
        """FAWOS works in imblearn Pipeline."""
        X, y = larger_binary_data

        pipe = ImbPipeline([
            ('sampler', FAWOS(
                sens_attr=sens_attr,
                priv_group=priv_group,
                random_state=42
            )),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)

    def test_heterogeneous_fos_in_pipeline(self, larger_binary_data, sens_attr):
        """HeterogeneousFOS works in imblearn Pipeline."""
        X, y = larger_binary_data

        pipe = ImbPipeline([
            ('sampler', HeterogeneousFOS(sens_attr=sens_attr, random_state=42)),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)

    def test_massaging_in_pipeline(self, larger_binary_data, sens_attr, priv_group):
        """Massaging works in imblearn Pipeline."""
        X, y = larger_binary_data

        pipe = ImbPipeline([
            ('sampler', Massaging(sens_attr=sens_attr, priv_group=priv_group)),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)

    def test_fairway_in_pipeline(self, larger_binary_data, sens_attr, priv_group):
        """FairwayRemover works in imblearn Pipeline."""
        X, y = larger_binary_data

        pipe = ImbPipeline([
            ('sampler', FairwayRemover(sens_attr=sens_attr, priv_group=priv_group)),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)

    def test_optimized_preprocessing_in_pipeline(self, sens_attr):
        """OptimizedPreprocessing works in imblearn Pipeline."""
        # Build a categorical dataset with integer-coded features
        # (algorithm requires discrete features; integers keep LogReg happy)
        rng = np.random.RandomState(42)
        n = 40
        X = pd.DataFrame({
            'age_cat': rng.choice([0, 1], n),       # 0=young, 1=old
            'group': np.array([0] * (n // 2) + [1] * (n // 2)),
        })
        y = np.concatenate([
            (rng.rand(n // 2) < 0.35).astype(int),
            (rng.rand(n // 2) < 0.65).astype(int),
        ])

        def distortion(old, new):
            cost = 0.0
            for k in old:
                if k != 'label' and old[k] != new[k]:
                    cost += 1.0
            if old['label'] != new['label']:
                cost += 2.0
            return cost

        pipe = ImbPipeline([
            ('sampler', OptimizedPreprocessing(
                sens_attr=sens_attr,
                features_to_transform=['age_cat'],
                distortion_fun=distortion,
                epsilon=0.5,
                random_state=42,
            )),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})


class TestCombinedPipelines:
    """Tests for combined transformer + resampler pipelines."""

    def test_transformer_then_resampler(self, larger_binary_data, sens_attr, priv_group):
        """Transformer followed by resampler in imblearn Pipeline."""
        X, y = larger_binary_data

        pipe = ImbPipeline([
            ('repair', GeometricFairnessRepair(
                sens_attr=sens_attr,
                repair_columns=['age', 'income'],
                lambda_param=0.5
            )),
            ('sampler', FairSmote(sens_attr=sens_attr, random_state=42)),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)

    def test_binarizer_then_resampler(self, larger_binary_data, sens_attr, priv_group):
        """IntersectionalBinarizer then resampler in imblearn Pipeline."""
        X, y = larger_binary_data

        pipe = ImbPipeline([
            ('binarizer', IntersectionalBinarizer(
                privileged_definition={'group': 1},
                group_col_name='is_privileged'
            )),
            ('sampler', FairOversampling(
                sens_attr=sens_attr,
                priv_group=priv_group,
                random_state=42
            )),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)


class TestPipelinePredictions:
    """Tests that pipelines produce valid predictions."""

    def test_pipeline_predict_proba(self, larger_binary_data, sens_attr):
        """Pipeline predict_proba returns valid probabilities."""
        X, y = larger_binary_data

        pipe = ImbPipeline([
            ('sampler', FairSmote(sens_attr=sens_attr, random_state=42)),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        proba = pipe.predict_proba(X)

        # Probabilities sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(len(X)))
        # Probabilities between 0 and 1
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_pipeline_score(self, larger_binary_data, sens_attr):
        """Pipeline score returns valid accuracy."""
        X, y = larger_binary_data

        pipe = ImbPipeline([
            ('sampler', FairSmote(sens_attr=sens_attr, random_state=42)),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])

        pipe.fit(X, y)
        score = pipe.score(X, y)

        assert 0 <= score <= 1

    def test_pipeline_with_different_classifiers(self, larger_binary_data, sens_attr):
        """Pipeline works with different classifier types."""
        X, y = larger_binary_data

        classifiers = [
            LogisticRegression(solver='liblinear', max_iter=200),
            DecisionTreeClassifier(max_depth=3, random_state=42),
        ]

        for clf in classifiers:
            pipe = ImbPipeline([
                ('sampler', FairSmote(sens_attr=sens_attr, random_state=42)),
                ('clf', clf)
            ])

            pipe.fit(X, y)
            preds = pipe.predict(X)

            assert len(preds) == len(X)


class TestFairMaskComplexPipelines:
    """Tests for FairMask in complex sklearn pipelines."""

    def test_fairmask_as_final_estimator(self, larger_binary_data, sens_attr):
        """FairMask works as final estimator in sklearn Pipeline."""
        X, y = larger_binary_data

        pipe = SklearnPipeline([
            ('clf', FairMask(
                estimator=LogisticRegression(solver='liblinear', max_iter=200),
                sens_attr=sens_attr,
                budget=3,
                random_state=42
            ))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)
        proba = pipe.predict_proba(X)

        assert len(preds) == len(X)
        assert proba.shape == (len(X), 2)
        assert set(preds).issubset({0, 1})

    def test_fairmask_with_transformer_preprocessing(self, larger_binary_data, sens_attr):
        """FairMask works with transformer preprocessing in sklearn Pipeline."""
        X, y = larger_binary_data

        pipe = SklearnPipeline([
            ('repair', GeometricFairnessRepair(
                sens_attr=sens_attr,
                repair_columns=['age', 'income'],
                lambda_param=0.5
            )),
            ('clf', FairMask(
                estimator=LogisticRegression(solver='liblinear', max_iter=200),
                sens_attr=sens_attr,
                budget=3,
                random_state=42
            ))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_fairmask_with_intersectional_binarizer(self, larger_binary_data, sens_attr):
        """FairMask works after IntersectionalBinarizer."""
        X, y = larger_binary_data

        pipe = SklearnPipeline([
            ('binarizer', IntersectionalBinarizer(
                privileged_definition={'group': 1},
                group_col_name='is_privileged'
            )),
            ('clf', FairMask(
                estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
                sens_attr=sens_attr,
                budget=3,
                random_state=42
            ))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)

    def test_fairmask_with_resampling_imblearn(self, larger_binary_data, sens_attr):
        """FairMask works after resampling in imblearn Pipeline."""
        X, y = larger_binary_data

        pipe = ImbPipeline([
            ('sampler', FairSmote(sens_attr=sens_attr, random_state=42)),
            ('clf', FairMask(
                estimator=LogisticRegression(solver='liblinear', max_iter=200),
                sens_attr=sens_attr,
                budget=3,
                random_state=42
            ))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_fairmask_complex_three_stage_pipeline(self, larger_binary_data, sens_attr):
        """FairMask in complex pipeline: transform -> resample -> classify."""
        X, y = larger_binary_data

        pipe = ImbPipeline([
            ('repair', GeometricFairnessRepair(
                sens_attr=sens_attr,
                repair_columns=['age', 'income'],
                lambda_param=0.3
            )),
            ('sampler', FairSmote(sens_attr=sens_attr, random_state=42)),
            ('clf', FairMask(
                estimator=LogisticRegression(solver='liblinear', max_iter=200),
                sens_attr=sens_attr,
                budget=5,
                random_state=42
            ))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)
        score = pipe.score(X, y)

        assert len(preds) == len(X)
        assert 0 <= score <= 1

    def test_fairmask_with_different_estimators(self, larger_binary_data, sens_attr):
        """FairMask works with different underlying estimators."""
        X, y = larger_binary_data
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB

        estimators = [
            LogisticRegression(solver='liblinear', max_iter=200),
            DecisionTreeClassifier(max_depth=5, random_state=42),
            RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
            GaussianNB(),
        ]

        for est in estimators:
            clf = FairMask(
                estimator=est,
                sens_attr=sens_attr,
                budget=3,
                random_state=42
            )
            clf.fit(X, y)
            preds = clf.predict(X)

            assert len(preds) == len(X)

    def test_fairmask_with_different_extrapolation_models(self, larger_binary_data, sens_attr):
        """FairMask works with different extrapolation models."""
        X, y = larger_binary_data
        from sklearn.ensemble import RandomForestClassifier

        extrapolation_models = [
            LogisticRegression(solver='liblinear', max_iter=200),
            DecisionTreeClassifier(max_depth=3, random_state=42),
            RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42),
        ]

        for ext_model in extrapolation_models:
            clf = FairMask(
                estimator=LogisticRegression(solver='liblinear', max_iter=200),
                sens_attr=sens_attr,
                budget=3,
                extrapolation_model=ext_model,
                random_state=42
            )
            clf.fit(X, y)
            preds = clf.predict(X)

            assert len(preds) == len(X)

    def test_fairmask_predict_proba_in_pipeline(self, larger_binary_data, sens_attr):
        """FairMask predict_proba returns valid probabilities in pipeline."""
        X, y = larger_binary_data

        pipe = SklearnPipeline([
            ('repair', GeometricFairnessRepair(
                sens_attr=sens_attr,
                repair_columns=['age', 'income'],
                lambda_param=0.5
            )),
            ('clf', FairMask(
                estimator=LogisticRegression(solver='liblinear', max_iter=200),
                sens_attr=sens_attr,
                budget=3,
                random_state=42
            ))
        ])

        pipe.fit(X, y)
        proba = pipe.predict_proba(X)

        # Probabilities sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(len(X)))
        # Probabilities between 0 and 1
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_fairmask_cross_validation(self, larger_binary_data, sens_attr):
        """FairMask works with cross-validation."""
        X, y = larger_binary_data

        clf = FairMask(
            estimator=LogisticRegression(solver='liblinear', max_iter=200),
            sens_attr=sens_attr,
            budget=3,
            random_state=42
        )

        # Run 3-fold CV
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_fairmask_pipeline_cross_validation(self, larger_binary_data, sens_attr):
        """FairMask in pipeline works with cross-validation."""
        X, y = larger_binary_data

        pipe = SklearnPipeline([
            ('repair', GeometricFairnessRepair(
                sens_attr=sens_attr,
                repair_columns=['age', 'income'],
                lambda_param=0.5
            )),
            ('clf', FairMask(
                estimator=LogisticRegression(solver='liblinear', max_iter=200),
                sens_attr=sens_attr,
                budget=3,
                random_state=42
            ))
        ])

        scores = cross_val_score(pipe, X, y, cv=3, scoring='accuracy')

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_fairmask_pipeline_grid_search(self, larger_binary_data, sens_attr):
        """FairMask supports grid search for hyperparameter tuning."""
        X, y = larger_binary_data

        # Use DecisionTreeClassifier for extrapolation as it handles single-class data better
        clf = FairMask(
            estimator=LogisticRegression(solver='liblinear', max_iter=200),
            sens_attr=sens_attr,
            extrapolation_model=DecisionTreeClassifier(max_depth=3, random_state=42),
            random_state=42
        )

        param_grid = {
            'budget': [3, 5],
        }

        # Use 3 folds to ensure enough data per fold
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X, y)

        assert grid_search.best_params_ is not None
        assert 0 <= grid_search.best_score_ <= 1

    def test_fairmask_pipeline_reproducibility(self, larger_binary_data, sens_attr):
        """FairMask pipeline produces reproducible results with same random_state."""
        X, y = larger_binary_data

        def make_pipeline():
            return SklearnPipeline([
                ('repair', GeometricFairnessRepair(
                    sens_attr=sens_attr,
                    repair_columns=['age', 'income'],
                    lambda_param=0.5
                )),
                ('clf', FairMask(
                    estimator=LogisticRegression(solver='liblinear', max_iter=200),
                    sens_attr=sens_attr,
                    budget=5,
                    random_state=42
                ))
            ])

        pipe1 = make_pipeline()
        pipe1.fit(X, y)
        preds1 = pipe1.predict(X)

        pipe2 = make_pipeline()
        pipe2.fit(X, y)
        preds2 = pipe2.predict(X)

        np.testing.assert_array_equal(preds1, preds2)

    def test_fairmask_chained_with_reweighing(self, larger_binary_data, sens_attr):
        """FairMask with ReweighingClassifier as underlying estimator."""
        X, y = larger_binary_data

        # Use ReweighingClassifier inside FairMask
        fairmask = FairMask(
            estimator=ReweighingClassifier(
                estimator=LogisticRegression(solver='liblinear', max_iter=200),
                sens_attr=sens_attr
            ),
            sens_attr=sens_attr,
            budget=3,
            random_state=42
        )

        fairmask.fit(X, y)
        preds = fairmask.predict(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_fairmask_multiple_fairness_steps(self, larger_binary_data, sens_attr, priv_group):
        """Pipeline with multiple fairness interventions ending with FairMask."""
        X, y = larger_binary_data

        pipe = ImbPipeline([
            ('repair', GeometricFairnessRepair(
                sens_attr=sens_attr,
                repair_columns=['age', 'income'],
                lambda_param=0.3
            )),
            ('binarizer', IntersectionalBinarizer(
                privileged_definition={'group': 1},
                group_col_name='is_privileged'
            )),
            ('sampler', FAWOS(
                sens_attr=sens_attr,
                priv_group=priv_group,
                random_state=42
            )),
            ('clf', FairMask(
                estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
                sens_attr=sens_attr,
                budget=3,
                random_state=42
            ))
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)
        score = pipe.score(X, y)

        assert len(preds) == len(X)
        assert 0 <= score <= 1

    def test_fairmask_budget_sensitivity(self, larger_binary_data, sens_attr):
        """Different budget values produce valid predictions."""
        X, y = larger_binary_data

        for budget in [1, 3, 10, 20]:
            clf = FairMask(
                estimator=LogisticRegression(solver='liblinear', max_iter=200),
                sens_attr=sens_attr,
                budget=budget,
                random_state=42
            )
            clf.fit(X, y)
            preds = clf.predict(X)

            assert len(preds) == len(X)
            assert len(clf.extrapolation_models_) == budget
            np.testing.assert_almost_equal(clf.model_weights_.sum(), 1.0)
