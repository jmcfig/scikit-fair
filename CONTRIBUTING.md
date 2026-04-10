# Contributing to scikit-fair

Thanks for your interest in contributing to scikit-fair! This guide covers the basics of how to get involved.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/scikit-fair.git
   cd scikit-fair
   ```
3. Create a conda environment and install dependencies:
   ```bash
   conda create -n scikit-fair python=3.10
   conda activate scikit-fair
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Preferentially run the test suite:
   ```bash
   pytest
   ```
4. Push to your fork and open a pull request against the `main` branch

## What Can I Contribute?

- **Bug reports**: open an issue describing the problem with a minimal reproducible example
- **New fairness methods**: add a new preprocessing method under `skfair/preprocessing/` and register it in the method registry
- **Additional datasets**: add a new loader under `skfair/datasets/` following the existing `load_*` pattern
- **New metrics**: add metric functions under `skfair/metrics/` and register them in the metric registry
- **Documentation**: improve docstrings, user guide pages, or example notebooks
- **Tests**: add tests for new features or to increase coverage of existing code
- **Examples**: create new example notebooks demonstrating how to use scikit-fair for different tasks
- And much, much more!

## Code Conventions

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- All public functions and classes should have NumPy-style docstrings
- New features should include tests under the corresponding `tests/` directory
- Maintain compatibility with the scikit-learn estimator API where applicable

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include a clear description of what the PR does and why
- Ensure all tests pass before requesting review
- Reference any related issues in the PR description

## Questions?

Feel free to open an issue on [GitHub](https://github.com/jmcfig/scikit-fair/issues) for any questions or discussion.
