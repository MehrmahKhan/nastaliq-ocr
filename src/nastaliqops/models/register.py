"""Model registry and the promotion gate.

A challenger is registered as a new version of the model nastaliq-ocr, and
it is only moved to the Production alias when it beats the current champion
on the golden set by the margin set in params.yaml. If it does not, the
champion stays where it is, which is the rollback behaviour the proposal
promises.
"""

import json
import sys

import mlflow

from mlflow.tracking import MlflowClient

from nastaliqops.config import Paths, load_params, mlflow_experiment, mlflow_uri


MODEL_NAME = "nastaliq-ocr"


def read_metrics():
    handle = open(Paths.reports / "metrics.json", "r", encoding="utf-8")
    metrics = json.load(handle)
    handle.close()
    return metrics


def champion_score(client, metric_name):
    """Return the metric of the version currently aliased Production."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, "Production")
    except Exception:
        return None

    run = client.get_run(version.run_id)
    if metric_name in run.data.metrics:
        return run.data.metrics[metric_name]
    return None


def main():
    params = load_params()
    promotion = params["promotion"]
    metric_name = promotion["metric"]

    mlflow.set_tracking_uri(mlflow_uri())
    mlflow.set_experiment(mlflow_experiment())
    client = MlflowClient()

    metrics = read_metrics()
    if "lora" not in metrics:
        print("no challenger metrics, nothing to register")
        sys.exit(1)

    challenger_value = metrics["lora"][metric_name]

    handle = open(Paths.artifacts / "last_run.json", "r", encoding="utf-8")
    last_run = json.load(handle)
    handle.close()

    model_uri = "runs:/" + last_run["run_id"] + "/adapter"
    version = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print("registered version " + str(version.version))

    client.set_model_version_tag(MODEL_NAME, version.version, "variant", "lora-adapter")
    client.set_model_version_tag(MODEL_NAME, version.version, "base", last_run["base_checkpoint"])
    client.set_model_version_tag(MODEL_NAME, version.version, metric_name, str(challenger_value))

    try:
        client.create_registered_model_alias(MODEL_NAME, "Challenger", version.version)
    except Exception:
        pass

    current = champion_score(client, "lora_" + metric_name)

    promote = False
    reason = ""

    if challenger_value > promotion["max_cer"]:
        reason = "challenger is above the absolute quality floor"
    elif current is None:
        promote = True
        reason = "no champion yet, this version becomes the first one"
    elif (current - challenger_value) >= promotion["min_improvement"]:
        promote = True
        reason = "challenger beats the champion by enough"
    else:
        reason = "challenger does not improve on the champion, keeping the current one"

    print(reason)

    if promote:
        client.set_registered_model_alias(MODEL_NAME, "Production", version.version)
        client.set_model_version_tag(MODEL_NAME, version.version, "promoted", "true")
        print("version " + str(version.version) + " is now Production")
    else:
        client.set_model_version_tag(MODEL_NAME, version.version, "promoted", "false")
        sys.exit(2)


if __name__ == "__main__":
    main()
