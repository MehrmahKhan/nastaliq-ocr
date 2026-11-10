"""Single place that loads params.yaml and the environment.

Every script imports from here so that the same config object is logged to
MLflow on every run. If a value is not in params.yaml it does not belong in
the code either.
"""

import os
import yaml

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"


def load_params():
    handle = open(PARAMS_PATH, "r", encoding="utf-8")
    params = yaml.safe_load(handle)
    handle.close()
    return params


def flatten_params(params, prefix=""):
    """Turn the nested dict into flat key/value pairs for mlflow.log_params.

    Written as a plain recursive loop on purpose, no comprehensions.
    """
    flat = {}
    for key in params:
        value = params[key]
        if prefix == "":
            full_key = str(key)
        else:
            full_key = prefix + "." + str(key)

        if isinstance(value, dict):
            inner = flatten_params(value, full_key)
            for inner_key in inner:
                flat[inner_key] = inner[inner_key]
        else:
            flat[full_key] = value
    return flat


class Paths:
    root = ROOT
    raw = ROOT / "data" / "raw"
    processed = ROOT / "data" / "processed"
    splits = ROOT / "data" / "splits"
    artifacts = ROOT / "artifacts"
    reports = ROOT / "reports"
    prompts = ROOT / "prompts"


def ensure_dirs():
    folders = [Paths.raw, Paths.processed, Paths.splits, Paths.artifacts, Paths.reports]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def env(name, default=""):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def mlflow_uri():
    """Where MLflow logs to.

    The default is a local sqlite file rather than the old './mlruns'
    directory, because recent MLflow refuses to write to the filesystem
    store and raises instead. Set MLFLOW_TRACKING_URI in .env to point at
    DagsHub or a local server.
    """
    default_db = ROOT / "mlruns" / "mlflow.db"
    default_db.parent.mkdir(parents=True, exist_ok=True)
    return env("MLFLOW_TRACKING_URI", "sqlite:///" + str(default_db))


def mlflow_experiment():
    return env("MLFLOW_EXPERIMENT", "nastaliq-ops")
