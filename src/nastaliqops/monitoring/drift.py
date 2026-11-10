"""Stage 7: drift detection with Evidently.

The training split is the reference. The stream split, which stands in for
live uploads until there are real ones, is the current window. Evidently
compares the two feature distributions and reports which columns drifted.

In production the current window is instead built from the predictions
table, so the same report runs over real traffic without any code change.
That path is wired through build_current_from_db below.

Outputs:
  reports/drift_report.html   the visual report, good for the demo
  reports/drift.json          the machine readable summary, tracked by dvc
"""

import json
import sys

import mlflow
import pandas as pd

from nastaliqops.config import (
    Paths,
    ensure_dirs,
    mlflow_experiment,
    mlflow_uri,
)
from nastaliqops.data.splits_io import read_split
from nastaliqops.monitoring.features import FEATURE_NAMES, features_for_file


DRIFT_SHARE_LIMIT = 0.3


def build_frame_from_split(split_name, limit):
    records = read_split(split_name)

    rows = []
    position = 0
    while position < len(records) and position < limit:
        record = records[position]
        position = position + 1

        image_path = Paths.processed / "images" / record["file_name"]
        features = features_for_file(image_path)
        if features is None:
            continue
        rows.append(features)

    return pd.DataFrame(rows)


def build_current_from_db(limit):
    """Build the current window from real served predictions instead.

    Used once the API has traffic. Returns an empty frame if the table is
    not there yet, and the caller falls back to the stream split.
    """
    try:
        from nastaliqops import review_store
    except Exception:
        return pd.DataFrame()

    try:
        connection = review_store.connect()
    except Exception:
        return pd.DataFrame()

    cursor = connection.cursor()
    cursor.execute(
        "SELECT image_path FROM predictions ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    connection.close()

    uploads = Paths.root / "data" / "uploads"
    collected = []
    for row in rows:
        features = features_for_file(uploads / row["image_path"])
        if features is None:
            continue
        collected.append(features)

    return pd.DataFrame(collected)


def _parse_modern_payload(payload):
    """Read the metric list produced by Evidently 0.7 and newer.

    That version returns a flat list of metrics. DriftedColumnsCount holds
    the totals, and one ValueDrift entry per column holds the p value, with
    the column name and the threshold written into metric_name.
    """
    summary = {}
    per_column = {}
    tested = 0
    drifted = 0
    share = 0.0

    metrics = payload.get("metrics", [])
    for entry in metrics:
        name = str(entry.get("metric_name", ""))
        value = entry.get("value")
        config = entry.get("config", {})

        if name.startswith("DriftedColumnsCount"):
            if isinstance(value, dict):
                drifted = int(value.get("count", 0))
                share = float(value.get("share", 0.0))
            continue

        if name.startswith("ValueDrift"):
            column_name = config.get("column")
            if column_name is None:
                continue

            threshold = config.get("threshold", 0.05)
            method = config.get("method", "")

            try:
                score = float(value)
            except (TypeError, ValueError):
                score = None

            if score is None:
                is_drifted = False
            else:
                is_drifted = score < float(threshold)

            per_column[str(column_name)] = {
                "drifted": is_drifted,
                "score": score,
                "test": str(method),
            }
            tested = tested + 1

    summary["columns_tested"] = tested
    summary["columns_drifted"] = drifted
    if tested > 0:
        summary["drift_share"] = float(drifted) / float(tested)
    else:
        summary["drift_share"] = share
    summary["by_column"] = per_column
    return summary


def _parse_legacy_payload(payload):
    """Read the nested result dict produced by Evidently 0.4 and 0.6."""
    drift_section = payload["metrics"][0]["result"]

    summary = {}
    summary["columns_tested"] = drift_section.get("number_of_columns", 0)
    summary["columns_drifted"] = drift_section.get("number_of_drifted_columns", 0)
    summary["drift_share"] = drift_section.get("share_of_drifted_columns", 0.0)

    per_column = {}
    by_columns = drift_section.get("drift_by_columns", {})
    for column_name in by_columns:
        entry = by_columns[column_name]
        per_column[column_name] = {
            "drifted": bool(entry.get("drift_detected", False)),
            "score": entry.get("drift_score", None),
            "test": entry.get("stattest_name", ""),
        }
    summary["by_column"] = per_column
    return summary


def _run_modern(reference, current):
    from evidently import Dataset, Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    run = report.run(Dataset.from_pandas(current), Dataset.from_pandas(reference))

    summary = _parse_modern_payload(run.dict())

    html_path = Paths.reports / "drift_report.html"
    run.save_html(str(html_path))
    handle = open(html_path, "r", encoding="utf-8")
    html = handle.read()
    handle.close()

    return summary, html


def _run_legacy(reference, current):
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset
    except ImportError:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    summary = _parse_legacy_payload(report.as_dict())
    return summary, report.get_html()


def run_report(reference, current):
    """Run Evidently and return (summary_dict, html_string).

    Evidently reorganised its API at 0.7, so the newer shape is tried first
    and the older one second. If neither works the caller falls back to a
    plain comparison rather than letting the stage pass silently.
    """
    attempts = [("evidently>=0.7", _run_modern), ("evidently<0.7", _run_legacy)]

    for label, runner in attempts:
        try:
            summary, html = runner(reference, current)
            print("drift computed with " + label)
            return summary, html
        except Exception as error:
            print(label + " path failed: " + str(error))

    return None, None


def fallback_report(reference, current):
    """A plain two sample comparison, used if Evidently cannot run.

    This keeps the pipeline stage honest rather than silently passing when
    the monitoring library is missing or has changed its API again.
    """
    summary = {}
    summary["columns_tested"] = 0
    summary["columns_drifted"] = 0
    per_column = {}

    for name in FEATURE_NAMES:
        if name not in reference.columns or name not in current.columns:
            continue

        reference_mean = float(reference[name].mean())
        reference_std = float(reference[name].std())
        current_mean = float(current[name].mean())

        if reference_std <= 0.0:
            score = 0.0
        else:
            score = abs(current_mean - reference_mean) / reference_std

        drifted = score > 0.5
        per_column[name] = {"drifted": drifted, "score": score, "test": "standardised_mean_shift"}

        summary["columns_tested"] = summary["columns_tested"] + 1
        if drifted:
            summary["columns_drifted"] = summary["columns_drifted"] + 1

    if summary["columns_tested"] == 0:
        summary["drift_share"] = 0.0
    else:
        summary["drift_share"] = float(summary["columns_drifted"]) / float(summary["columns_tested"])

    summary["by_column"] = per_column
    return summary


def fallback_html(summary):
    """A plain table, so the dvc output always exists even without Evidently."""
    rows = ""
    per_column = summary.get("by_column", {})
    for name in per_column:
        entry = per_column[name]
        if entry["drifted"]:
            verdict = "drifted"
        else:
            verdict = "stable"
        score = entry["score"]
        if score is None:
            score_text = "n/a"
        else:
            score_text = str(round(float(score), 4))
        rows = rows + "<tr><td>" + name + "</td><td>" + score_text + "</td><td>" + verdict + "</td></tr>"

    html = "<!doctype html><html><head><meta charset='utf-8'>"
    html = html + "<title>Drift report</title>"
    html = html + "<style>body{font-family:system-ui,sans-serif;margin:40px;}"
    html = html + "table{border-collapse:collapse;}td,th{border:1px solid #ccc;padding:8px 14px;}"
    html = html + "</style></head><body>"
    html = html + "<h1>Drift report</h1>"
    html = html + "<p>Detector: " + str(summary.get("detector", "fallback")) + ". "
    html = html + "Reference rows: " + str(summary.get("reference_rows", 0)) + ". "
    html = html + "Current rows: " + str(summary.get("current_rows", 0)) + ".</p>"
    html = html + "<p>Drift share: " + str(round(summary.get("drift_share", 0.0), 3)) + "</p>"
    html = html + "<table><tr><th>feature</th><th>score</th><th>verdict</th></tr>"
    html = html + rows + "</table></body></html>"
    return html


def main():
    ensure_dirs()

    limit = 500
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])

    reference = build_frame_from_split("train", limit)

    current = build_current_from_db(limit)
    source = "served_predictions"
    if len(current) < 30:
        current = build_frame_from_split("stream", limit)
        source = "stream_split"

    print("reference rows: " + str(len(reference)))
    print("current rows: " + str(len(current)) + " from " + source)

    if len(reference) == 0 or len(current) == 0:
        print("not enough data to compare, run the data stages first")
        sys.exit(1)

    summary, html = run_report(reference, current)
    used = "evidently"
    if summary is None:
        summary = fallback_report(reference, current)
        used = "fallback"

    summary["source"] = source
    summary["detector"] = used
    summary["reference_rows"] = len(reference)
    summary["current_rows"] = len(current)
    summary["drift_share_limit"] = DRIFT_SHARE_LIMIT
    summary["retraining_recommended"] = summary["drift_share"] > DRIFT_SHARE_LIMIT

    out_json = Paths.reports / "drift.json"
    handle = open(out_json, "w", encoding="utf-8")
    handle.write(json.dumps(summary, indent=2))
    handle.close()

    # the html is a declared dvc output, so it has to exist either way
    if html is None:
        html = fallback_html(summary)

    out_html = Paths.reports / "drift_report.html"
    handle = open(out_html, "w", encoding="utf-8")
    handle.write(html)
    handle.close()
    print("wrote " + str(out_html))

    try:
        mlflow.set_tracking_uri(mlflow_uri())
        mlflow.set_experiment(mlflow_experiment())
        mlflow.start_run(run_name="drift-check")
        mlflow.log_metric("drift_share", summary["drift_share"])
        mlflow.log_metric("columns_drifted", summary["columns_drifted"])
        mlflow.log_artifact(str(out_json))
        mlflow.log_artifact(str(out_html))
        mlflow.end_run()
    except Exception as error:
        print("could not log to mlflow: " + str(error))

    print("drift share: " + str(round(summary["drift_share"], 3)) +
          " using " + used)

    if summary["retraining_recommended"]:
        print("drift above the limit, retraining is recommended")


if __name__ == "__main__":
    main()
