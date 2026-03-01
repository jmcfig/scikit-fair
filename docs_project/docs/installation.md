# Installation

## Requirements

- Python >= 3.9
- numpy >= 1.22
- pandas >= 1.5
- scikit-learn >= 1.3, < 2.0
- imbalanced-learn >= 0.12
- cvxpy >= 1.3 (required for `OptimizedPreprocessing`)

## From source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-org/scikit-fair.git
cd scikit-fair
pip install -e .
```

## Verify

```python
import skfair
from skfair.preprocessing import Massaging
from skfair.metrics import disparate_impact
from skfair.datasets import load_adult
```
