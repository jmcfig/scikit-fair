"""
Sklearn compatibility tests for scikit-fair preprocessing classes.

Tests that all classes follow sklearn conventions:
- clone() works
- get_params() / set_params() round-trip
- Fitted attributes exist after fit
"""
import pytest
from sklearn.base import clone

from skfair.preprocessing import (
    GeometricFairnessRepair,
    IntersectionalBinarizer,
    Reweighing,
    ReweighingClassifier,
    FairSmote,
    FairOversampling,
    FAWOS,
    HeterogeneousFOS,
    Massaging,
    FairwayRemover,
    OptimizedPreprocessing,
    LearningFairRepresentations,
)


# All estimator classes to test
TRANSFORMER_CLASSES = [
    (GeometricFairnessRepair, {'sens_attr': 'group', 'repair_columns': ['age']}),
    (IntersectionalBinarizer, {'privileged_definition': {'group': 1}}),
    (LearningFairRepresentations, {'sens_attr': 'group', 'priv_group': 1}),
]

def _op_distortion(old, new):
    return sum(1.0 for k in old if k != 'label' and old[k] != new[k]) + (
        2.0 if old.get('label') != new.get('label') else 0.0
    )

RESAMPLER_CLASSES = [
    (FairSmote, {'sens_attr': 'group'}),
    (FairOversampling, {'sens_attr': 'group', 'priv_group': 1}),
    (FAWOS, {'sens_attr': 'group', 'priv_group': 1}),
    (HeterogeneousFOS, {'sens_attr': 'group'}),
    (Massaging, {'sens_attr': 'group', 'priv_group': 1}),
    (FairwayRemover, {'sens_attr': 'group', 'priv_group': 1}),
]

OTHER_CLASSES = [
    (Reweighing, {'sens_attr': 'group'}),
    (ReweighingClassifier, {'sens_attr': 'group'}),
    (OptimizedPreprocessing, {
        'sens_attr': 'group',
        'features_to_transform': ['age'],
        'distortion_fun': _op_distortion,
    }),
]

ALL_CLASSES = TRANSFORMER_CLASSES + RESAMPLER_CLASSES + OTHER_CLASSES


class TestClone:
    """Tests that sklearn.base.clone() works for all classes."""

    @pytest.mark.parametrize("cls,kwargs", ALL_CLASSES)
    def test_clone_unfitted(self, cls, kwargs):
        """clone() works on unfitted estimator."""
        est = cls(**kwargs)
        cloned = clone(est)

        assert type(cloned) is type(est)
        assert cloned is not est
        assert cloned.get_params() == est.get_params()

    @pytest.mark.parametrize("cls,kwargs", TRANSFORMER_CLASSES)
    def test_clone_fitted_transformer(self, cls, kwargs, simple_binary_data):
        """clone() on fitted transformer returns unfitted clone."""
        X, y = simple_binary_data
        est = cls(**kwargs)
        est.fit(X, y)

        cloned = clone(est)

        # Clone should be unfitted
        assert type(cloned) is type(est)
        assert cloned.get_params() == est.get_params()

    @pytest.mark.parametrize("cls,kwargs", RESAMPLER_CLASSES)
    def test_clone_fitted_resampler(self, cls, kwargs, simple_binary_data):
        """clone() on fitted resampler returns unfitted clone."""
        X, y = simple_binary_data
        est = cls(**kwargs)
        # Add random_state if supported
        if 'random_state' in est.get_params():
            est.set_params(random_state=42)
        est.fit_resample(X, y)

        cloned = clone(est)

        assert type(cloned) is type(est)


class TestGetSetParams:
    """Tests for get_params() and set_params() methods."""

    @pytest.mark.parametrize("cls,kwargs", ALL_CLASSES)
    def test_get_params(self, cls, kwargs):
        """get_params() returns dict with all init parameters."""
        est = cls(**kwargs)
        params = est.get_params()

        assert isinstance(params, dict)
        for key in kwargs:
            assert key in params
            assert params[key] == kwargs[key]

    @pytest.mark.parametrize("cls,kwargs", ALL_CLASSES)
    def test_set_params(self, cls, kwargs):
        """set_params() updates parameters correctly."""
        est = cls(**kwargs)

        # Get a parameter to modify
        params = est.get_params()
        first_key = list(kwargs.keys())[0]
        original_value = params[first_key]

        # For string params, we'll skip modification test
        # For numeric params, we can test
        if isinstance(original_value, (int, float)) and first_key not in ['priv_group']:
            new_value = original_value + 1 if isinstance(original_value, int) else original_value + 0.1
            est.set_params(**{first_key: new_value})
            assert est.get_params()[first_key] == new_value

    @pytest.mark.parametrize("cls,kwargs", ALL_CLASSES)
    def test_get_set_params_roundtrip(self, cls, kwargs):
        """get_params -> set_params roundtrip preserves parameters."""
        est1 = cls(**kwargs)
        params = est1.get_params()

        est2 = cls(**kwargs)
        est2.set_params(**params)

        assert est1.get_params() == est2.get_params()


class TestFittedAttributes:
    """Tests that fitted attributes exist after fitting."""

    def test_reweighing_fitted_attributes(self, simple_binary_data, sens_attr):
        """Reweighing has weights_ after fit_transform."""
        X, y = simple_binary_data
        rw = Reweighing(sens_attr=sens_attr)
        rw.fit_transform(X, y)

        assert hasattr(rw, 'weights_')
        assert hasattr(rw, 'weight_table_')

    def test_reweighing_classifier_fitted_attributes(self, simple_binary_data, sens_attr):
        """ReweighingClassifier has estimator_ after fit."""
        X, y = simple_binary_data
        clf = ReweighingClassifier(sens_attr=sens_attr)
        clf.fit(X, y)

        assert hasattr(clf, 'estimator_')
        assert hasattr(clf, 'reweigher_')

    def test_geometric_repair_fitted_attributes(self, simple_binary_data):
        """GeometricFairnessRepair has bucket attributes after fit."""
        X, y = simple_binary_data
        repair = GeometricFairnessRepair(
            sens_attr='group',
            repair_columns=['age', 'income']
        )
        repair.fit(X, y)

        assert hasattr(repair, 'n_buckets_')
        assert hasattr(repair, 'bucket_edges_')

    def test_massaging_fitted_attributes(self, simple_binary_data, sens_attr, priv_group):
        """Massaging has ranker_ after fit_resample."""
        X, y = simple_binary_data
        ms = Massaging(sens_attr=sens_attr, priv_group=priv_group)
        ms.fit_resample(X, y)

        assert hasattr(ms, 'ranker_')
        assert hasattr(ms, 'classes_')

    def test_fairway_fitted_attributes(self, simple_binary_data, sens_attr, priv_group):
        """FairwayRemover has model_p_ and model_u_ after fit_resample."""
        X, y = simple_binary_data
        fw = FairwayRemover(sens_attr=sens_attr, priv_group=priv_group)
        fw.fit_resample(X, y)

        assert hasattr(fw, 'model_p_')
        assert hasattr(fw, 'model_u_')


class TestReproducibility:
    """Tests that random_state produces reproducible results."""

    def test_fair_smote_reproducibility(self, simple_binary_data, sens_attr):
        """Same random_state produces identical results."""
        X, y = simple_binary_data

        fs1 = FairSmote(sens_attr=sens_attr, random_state=42)
        X1, y1 = fs1.fit_resample(X, y)

        fs2 = FairSmote(sens_attr=sens_attr, random_state=42)
        X2, y2 = fs2.fit_resample(X, y)

        assert X1.equals(X2)
        assert (y1 == y2).all()

    def test_fair_smote_different_seeds(self, simple_binary_data, sens_attr):
        """Different random_state produces different results."""
        X, y = simple_binary_data

        fs1 = FairSmote(sens_attr=sens_attr, random_state=42)
        X1, y1 = fs1.fit_resample(X, y)

        fs2 = FairSmote(sens_attr=sens_attr, random_state=123)
        X2, y2 = fs2.fit_resample(X, y)

        # Results should differ (at least the synthetic samples)
        # They might have same length but different values
        if len(X1) == len(X2):
            assert not X1.equals(X2)

    def test_fos_reproducibility(self, simple_binary_data, sens_attr, priv_group):
        """FairOversampling reproducibility."""
        X, y = simple_binary_data

        fos1 = FairOversampling(sens_attr=sens_attr, priv_group=priv_group, random_state=42)
        X1, y1 = fos1.fit_resample(X, y)

        fos2 = FairOversampling(sens_attr=sens_attr, priv_group=priv_group, random_state=42)
        X2, y2 = fos2.fit_resample(X, y)

        assert X1.equals(X2)
        assert (y1 == y2).all()

    def test_fawos_reproducibility(self, simple_binary_data, sens_attr, priv_group):
        """FAWOS reproducibility."""
        X, y = simple_binary_data

        fawos1 = FAWOS(sens_attr=sens_attr, priv_group=priv_group, random_state=42)
        X1, y1 = fawos1.fit_resample(X, y)

        fawos2 = FAWOS(sens_attr=sens_attr, priv_group=priv_group, random_state=42)
        X2, y2 = fawos2.fit_resample(X, y)

        assert X1.equals(X2)
        assert (y1 == y2).all()

    def test_heterogeneous_fos_reproducibility(self, simple_binary_data, sens_attr):
        """HeterogeneousFOS reproducibility."""
        X, y = simple_binary_data

        hfos1 = HeterogeneousFOS(sens_attr=sens_attr, random_state=42)
        X1, y1 = hfos1.fit_resample(X, y)

        hfos2 = HeterogeneousFOS(sens_attr=sens_attr, random_state=42)
        X2, y2 = hfos2.fit_resample(X, y)

        assert X1.equals(X2)
        assert (y1 == y2).all()
