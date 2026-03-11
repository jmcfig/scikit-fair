"""
XML configuration parser for the Experiment class.

Reads an XML file or string and produces the configuration dict consumed
by ``Experiment.__init__``.
"""

import os
import xml.etree.ElementTree as ET


def _auto_cast(value):
    """Cast an XML attribute string to a Python scalar.

    ``"true"``/``"false"`` → bool, integers → int, floats → float,
    otherwise the original string is returned.
    """
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_experiment_xml(source):
    """Parse an experiment XML configuration.

    Parameters
    ----------
    source : str
        A file path **or** an XML string (detected by the presence of ``<``).

    Returns
    -------
    dict
        Keys: ``datasets``, ``methods``, ``classifiers``, ``cv``, ``audit``,
        ``metrics``.  Missing sections produce sensible defaults (empty lists
        or dicts).
    """
    if "<" in source:
        root = ET.fromstring(source)
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"XML config not found: {source}")
        tree = ET.parse(source)
        root = tree.getroot()

    config = {
        "datasets": [],
        "methods": [],
        "classifiers": [],
        "cv": {},
        "audit": {},
        "metrics": [],
    }

    # --- datasets ---
    ds_node = root.find("datasets")
    if ds_node is not None:
        for ds in ds_node.findall("dataset"):
            entry = {}
            for k, v in ds.attrib.items():
                entry[k] = _auto_cast(v)
            config["datasets"].append(entry)

    # --- methods ---
    m_node = root.find("methods")
    if m_node is not None:
        for m in m_node.findall("method"):
            entry = {}
            for k, v in m.attrib.items():
                entry[k] = _auto_cast(v)
            config["methods"].append(entry)

    # --- classifiers ---
    clf_node = root.find("classifiers")
    if clf_node is not None:
        for clf in clf_node.findall("classifier"):
            entry = {}
            for k, v in clf.attrib.items():
                entry[k] = _auto_cast(v)
            config["classifiers"].append(entry)

    # --- cv ---
    cv_node = root.find("cv")
    if cv_node is not None:
        for k, v in cv_node.attrib.items():
            config["cv"][k] = _auto_cast(v)

    # --- audit ---
    audit_node = root.find("audit")
    if audit_node is not None:
        for k, v in audit_node.attrib.items():
            config["audit"][k] = _auto_cast(v)

    # --- metrics ---
    metrics_node = root.find("metrics")
    if metrics_node is not None:
        for met in metrics_node.findall("metric"):
            entry = {}
            for k, v in met.attrib.items():
                entry[k] = _auto_cast(v)
            config["metrics"].append(entry)

    return config
