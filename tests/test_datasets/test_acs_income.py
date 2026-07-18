import numpy as np
import pandas as pd
import pytest
from sklearn.utils import Bunch

from skfair.datasets import fetch_acs_income


def _fake_pums_frame(n=60):
    """Synthetic frame mimicking the raw OpenML ACSIncome layout."""
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "AGEP": rng.randint(18, 90, n).astype(float),
        "COW": rng.randint(1, 9, n).astype(float),
        "SCHL": rng.randint(1, 25, n).astype(float),
        "MAR": rng.randint(1, 6, n).astype(float),
        "OCCP": rng.randint(10, 9800, n).astype(float),
        "POBP": rng.randint(1, 500, n).astype(float),
        "RELP": rng.randint(0, 18, n).astype(float),
        "WKHP": rng.randint(1, 99, n).astype(float),
        "SEX": rng.choice([1.0, 2.0], n),
        "RAC1P": rng.choice([1.0, 2.0, 6.0, 8.0], n),
        "ST": rng.randint(1, 57, n).astype(float),
        "PINCP": rng.randint(100, 200000, n).astype(float),
    })


@pytest.fixture
def mocked_openml(monkeypatch):
    frame = _fake_pums_frame()
    monkeypatch.setattr(
        "skfair.datasets._acs_income.fetch_openml",
        lambda **kwargs: Bunch(frame=frame.copy()),
    )
    return frame


def test_preprocessed_default(mocked_openml):
    X, y = fetch_acs_income()

    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(mocked_openml)
    assert "ST" not in X.columns
    assert "PINCP" not in X.columns

    # Sensitive columns kept intact: SEX binarised, RAC1P multi-valued codes
    assert set(np.unique(X["SEX"])) <= {0, 1}
    assert len(set(np.unique(X["SEX"]))) == 2
    assert X["RAC1P"].nunique() > 2
    expected_race = mocked_openml["RAC1P"].astype(int).to_numpy()
    np.testing.assert_array_equal(X["RAC1P"].to_numpy(), expected_race)

    # Binary target from the $50K threshold
    expected_y = (mocked_openml["PINCP"] > 50_000).astype(int).to_numpy()
    np.testing.assert_array_equal(np.asarray(y), expected_y)

    # Remaining features standardized
    assert abs(X["AGEP"].mean()) < 1e-9
    assert abs(X["WKHP"].std() - 1.0) < 1e-6


def test_raw(mocked_openml):
    X, y = fetch_acs_income(preprocessed=False)

    assert "ST" in X.columns
    assert "PINCP" not in X.columns
    pd.testing.assert_series_equal(y, mocked_openml["PINCP"], check_names=True)
    pd.testing.assert_frame_equal(X, mocked_openml.drop(columns=["PINCP"]))


def test_subsample(mocked_openml):
    X, y = fetch_acs_income(subsample=10, random_state=42)

    assert len(X) == 10
    assert len(y) == 10
    assert isinstance(X.index, pd.RangeIndex)

    X2, y2 = fetch_acs_income(subsample=10, random_state=42)
    pd.testing.assert_frame_equal(X, X2)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(y2))


def test_bunch_return(mocked_openml):
    bunch = fetch_acs_income(return_X_y=False)

    assert set(bunch.keys()) >= {"data", "target", "frame", "feature_names", "DESCR"}
    assert list(bunch.data.columns) == bunch.feature_names
    assert bunch.frame.shape[1] == bunch.data.shape[1] + 1
    assert "OpenML" in bunch.DESCR


def test_as_frame_false(mocked_openml):
    X, y = fetch_acs_income(as_frame=False)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0] == len(mocked_openml)


@pytest.mark.network
def test_real_fetch():
    """Real OpenML download; run with ``pytest -m network``."""
    X, y = fetch_acs_income()

    assert X.shape[0] == 1_664_500
    assert set(np.unique(y)) == {0, 1}
    assert set(np.unique(X["SEX"])) == {0, 1}
    assert X["RAC1P"].nunique() > 2
