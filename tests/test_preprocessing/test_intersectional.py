import pandas as pd
import numpy as np
import pytest
from skfair.preprocessing import IntersectionalBinarizer


@pytest.fixture
def sample_data():
    """
    Creates a simple DataFrame covering various sensitive attribute combinations.
    sex: 1 (Male), 0 (Female)
    race: 1 (White), 0 (Black)
    age: Used for numerical threshold testing
    """
    data = {
        'sex': [1, 0, 1, 0, 1, 0],
        'race': [1, 1, 0, 0, 1, 0],
        'age': [30, 45, 22, 60, 65, 70],
        'feature_A': np.random.rand(6)
    }
    return pd.DataFrame(data)


def test_numerical_threshold_ge(sample_data):
    """
    Tests privilege definition using '>=' operator on a numerical attribute.
    Privileged: age >= 60
    """
    definition = {'age': {'>=': 60}}
    binarizer = IntersectionalBinarizer(
        privileged_definition=definition,
        group_col_name='is_privileged'
    )

    X_transformed = binarizer.fit_transform(sample_data.copy())

    # age values: [30, 45, 22, 60, 65, 70]
    # Expected output array (60, 65, 70 should be 1):
    expected_privileged = np.array([0, 0, 0, 1, 1, 1])

    assert (X_transformed['is_privileged'].values == expected_privileged).all()


def test_inequality_operator_ne(sample_data):
    """
    Tests privilege definition using '!=' operator.
    Privileged: race != 1 (i.e., race == 0)
    """
    definition = {'race': {'!=': 1}}
    binarizer = IntersectionalBinarizer(
        privileged_definition=definition,
        group_col_name='is_privileged'
    )

    X_transformed = binarizer.fit_transform(sample_data.copy())

    # race values: [1, 1, 0, 0, 1, 0]
    # Expected output array (0 should be 1):
    expected_privileged = np.array([0, 0, 1, 1, 0, 1])

    assert (X_transformed['is_privileged'].values == expected_privileged).all()


def test_complex_mixed_operators(sample_data):
    """
    Tests a complex rule combining equality, OR, and numerical operators.
    Privileged: (sex=1 AND age > 50) OR (race=0 AND age <= 30)
    """
    definition = [
        {'sex': 1, 'age': {'>': 50}},
        {'race': 0, 'age': {'<=': 30}}
    ]
    binarizer = IntersectionalBinarizer(
        privileged_definition=definition,
        group_col_name='is_privileged'
    )

    X_transformed = binarizer.fit_transform(sample_data.copy())

    # Analysis:
    # R0: M, W, 30 -> No
    # R1: F, W, 45 -> No
    # R2: M, B, 22 -> Yes (Rule 2: race=0 AND age<=30)
    # R3: F, B, 60 -> No (Rule 2 fails age<=30)
    # R4: M, W, 65 -> Yes (Rule 1: sex=1 AND age>50)
    # R5: F, B, 70 -> No
    expected_privileged = np.array([0, 0, 1, 0, 1, 0])

    assert (X_transformed['is_privileged'].values == expected_privileged).all()


def test_invalid_operator_error(sample_data):
    """
    Tests that a ValueError is raised if an unsupported operator is used.
    """
    definition = {'age': {'~': 50}}  # '~' is unsupported
    binarizer = IntersectionalBinarizer(
        privileged_definition=definition,
        group_col_name='is_privileged'
    )

    with pytest.raises(ValueError, match="Unsupported operator '~'"):
        binarizer.fit_transform(sample_data.copy())
