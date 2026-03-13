"""
YAML configuration parser for the Experiment class.

Reads a YAML file or string and produces the configuration dict consumed
by ``Experiment.__init__``.
"""

import os

import yaml


def parse_experiment_config(source):
    """Parse an experiment YAML configuration.

    Parameters
    ----------
    source : str
        A file path **or** a YAML string (detected by the presence of ``\\n``).

    Returns
    -------
    dict
        Keys: ``datasets``, ``methods``, ``classifiers``, ``cv``, ``audit``,
        ``metrics``.  Missing sections produce sensible defaults (empty lists
        or dicts).
    """
    if "\n" in source:
        raw = yaml.safe_load(source)
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Config file not found: {source}")
        with open(source) as f:
            raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    config = {
        "datasets": [],
        "methods": [],
        "classifiers": [],
        "cv": {},
        "audit": {},
        "metrics": [],
    }

    # --- datasets ---
    for item in raw.get("datasets", []):
        if isinstance(item, str):
            config["datasets"].append({"name": item})
        elif isinstance(item, dict):
            config["datasets"].append(item)

    # --- methods ---
    for item in raw.get("methods", []):
        if isinstance(item, str):
            config["methods"].append({"name": item})
        elif isinstance(item, dict):
            config["methods"].append(item)

    # --- classifiers ---
    for item in raw.get("classifiers", []):
        if isinstance(item, dict):
            config["classifiers"].append(item)

    # --- cv ---
    cv = raw.get("cv", {})
    if isinstance(cv, dict):
        config["cv"] = cv

    # --- audit ---
    audit = raw.get("audit", {})
    if isinstance(audit, dict):
        config["audit"] = audit

    # --- metrics ---
    for item in raw.get("metrics", []):
        if isinstance(item, str):
            config["metrics"].append({"name": item})
        elif isinstance(item, dict):
            config["metrics"].append(item)

    return config
