"""Stage 2: data validation gate, enforced with Pandera.

A training run should refuse to start on a bad dataset. This stage builds a
dataframe describing every row, including the image dimensions, runs the
Pandera schema over it and writes reports/data_validation.json. If the
failure rate is above the allowed limit the stage exits non zero, which
stops dvc repro and fails the CI job.

The schema itself lives in schema.py so it can be tested without images.
"""

import json
import sys

import pandas as pd

from PIL import Image

from nastaliqops.config import Paths, ensure_dirs
from nastaliqops.data.schema import MAX_FAILURE_RATE, validate_frame
from nastaliqops.utils.urdu import normalise


def describe_image(image_path):
    """Return width, height and whether the file could be opened at all."""
    if not image_path.exists():
        return 0, 0, False

    try:
        image = Image.open(image_path)
        width, height = image.size
        return int(width), int(height), True
    except Exception:
        return 0, 0, False


def build_frame(manifest_path):
    handle = open(manifest_path, "r", encoding="utf-8")
    lines = handle.readlines()
    handle.close()

    rows = []
    unreadable = []

    for line in lines:
        line = line.strip()
        if line == "":
            continue

        record = json.loads(line)
        image_path = Paths.raw / record.get("rel_path", "images/" + record["file_name"])
        width, height, readable = describe_image(image_path)

        if not readable:
            unreadable.append(record["id"])

        if height > 0:
            aspect = float(width) / float(height)
        else:
            aspect = 0.0

        row = {}
        row["id"] = int(record["id"])
        row["file_name"] = str(record["file_name"])
        row["text"] = normalise(record.get("text", ""))
        row["width"] = width
        row["height"] = height
        row["aspect_ratio"] = aspect
        rows.append(row)

    return pd.DataFrame(rows), unreadable


def main():
    ensure_dirs()

    manifest_path = Paths.raw / "manifest.jsonl"
    frame, unreadable = build_frame(manifest_path)

    total = len(frame)
    failed_ids, reasons = validate_frame(frame)

    # a file that would not open is a failure the schema cannot see, since
    # it only sees the zeros we wrote for its dimensions
    for bad_id in unreadable:
        if bad_id not in failed_ids:
            failed_ids.append(bad_id)
    if len(unreadable) > 0:
        reasons["image_unreadable"] = len(unreadable)

    failed = len(failed_ids)
    if total == 0:
        failure_rate = 1.0
    else:
        failure_rate = float(failed) / float(total)

    report = {}
    report["validator"] = "pandera"
    report["total_rows"] = total
    report["failed_rows"] = failed
    report["failure_rate"] = failure_rate
    report["max_allowed_failure_rate"] = MAX_FAILURE_RATE
    report["reasons"] = reasons
    report["rejected_ids"] = failed_ids
    report["passed"] = failure_rate <= MAX_FAILURE_RATE

    out_path = Paths.reports / "data_validation.json"
    handle = open(out_path, "w", encoding="utf-8")
    handle.write(json.dumps(report, ensure_ascii=False, indent=2))
    handle.close()

    print("validated " + str(total) + " rows, " + str(failed) + " failed")
    for reason in reasons:
        print("  " + reason + ": " + str(reasons[reason]))

    if not report["passed"]:
        print("failure rate above the allowed limit, stopping the pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()
