import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def simple_binary_data():
    """
    Minimal dataset for testing fairness-aware preprocessors.

    20 samples with:
    - 2 numeric features (age, income)
    - 1 sensitive attribute (group: 0 or 1)
    - Binary labels with imbalance across groups

    Distribution:
    - Group 0 (unprivileged): 10 samples, 3 positive, 7 negative
    - Group 1 (privileged): 10 samples, 7 positive, 3 negative
    """
    data = {
        'age': [25, 30, 35, 40, 45, 50, 55, 60, 28, 33,
                22, 27, 32, 37, 42, 47, 52, 57, 38, 43],
        'income': [30000, 40000, 50000, 60000, 70000,
                   35000, 45000, 55000, 65000, 75000,
                   32000, 42000, 52000, 62000, 72000,
                   38000, 48000, 58000, 68000, 78000],
        'group': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    X = pd.DataFrame(data)

    # Labels: privileged (group=1) has higher positive rate
    y = np.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 0,   # group 0: 3 pos
                  1, 1, 0, 1, 1, 1, 0, 1, 0, 1])  # group 1: 7 pos

    return X, y


@pytest.fixture
def sens_attr():
    """Protected attribute column name."""
    return 'group'


@pytest.fixture
def priv_group():
    """Value representing the privileged group."""
    return 1


@pytest.fixture
def pos_label():
    """Value representing the positive/favorable outcome."""
    return 1


@pytest.fixture
def larger_binary_data():
    """
    Larger dataset for pipeline integration tests.

    100 samples with:
    - 2 numeric features (age, income)
    - 1 sensitive attribute (group: 0 or 1)
    - Binary labels with imbalance across groups

    Distribution:
    - Group 0 (unprivileged): 50 samples, 15 positive, 35 negative
    - Group 1 (privileged): 50 samples, 35 positive, 15 negative
    """
    np.random.seed(42)
    n_per_group = 50

    # Group 0
    age_0 = np.random.randint(20, 65, n_per_group)
    income_0 = np.random.randint(25000, 80000, n_per_group)
    y_0 = np.array([1] * 15 + [0] * 35)
    np.random.shuffle(y_0)

    # Group 1
    age_1 = np.random.randint(20, 65, n_per_group)
    income_1 = np.random.randint(25000, 80000, n_per_group)
    y_1 = np.array([1] * 35 + [0] * 15)
    np.random.shuffle(y_1)

    X = pd.DataFrame({
        'age': np.concatenate([age_0, age_1]),
        'income': np.concatenate([income_0, income_1]),
        'group': [0] * n_per_group + [1] * n_per_group,
    })
    y = np.concatenate([y_0, y_1])

    return X, y


@pytest.fixture
def balanced_data():
    """
    Already balanced dataset - same positive rate across groups.

    20 samples with:
    - Group 0: 10 samples, 5 positive, 5 negative
    - Group 1: 10 samples, 5 positive, 5 negative
    """
    data = {
        'age': [25, 30, 35, 40, 45, 50, 55, 60, 28, 33,
                22, 27, 32, 37, 42, 47, 52, 57, 38, 43],
        'income': [30000, 40000, 50000, 60000, 70000,
                   35000, 45000, 55000, 65000, 75000,
                   32000, 42000, 52000, 62000, 72000,
                   38000, 48000, 58000, 68000, 78000],
        'group': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    X = pd.DataFrame(data)

    # Balanced: same positive rate in both groups
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0,   # group 0: 5 pos, 5 neg
                  1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # group 1: 5 pos, 5 neg

    return X, y


@pytest.fixture
def minimal_data():
    """
    Minimum viable dataset for k_neighbors=5.

    24 samples: 6 per subgroup (class x group).
    Just enough for KNN with k=5.
    """
    data = {
        'age': list(range(20, 44)),
        'income': list(range(30000, 54000, 1000)),
        'group': [0] * 12 + [1] * 12,
    }
    X = pd.DataFrame(data)

    # 6 samples per (group, class) combination
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,   # group 0: 6 neg, 6 pos
                  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # group 1: 6 neg, 6 pos

    return X, y
