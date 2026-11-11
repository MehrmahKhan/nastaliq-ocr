"""Stage 4: split into train, val, golden and stream.

The split is deliberately not random across everything:

  golden  real scanned lines only. Nothing trains on it and promotion is
          decided on it alone, so it has to be the hardest and most honest
          data available.
  stream  the deliberately degraded synthetic lines, standing in for live
          uploads until the service has real traffic. Because they were
          rendered with a higher severity than the training set, the drift
          monitor has something genuine to detect.
  train   clean synthetic lines.
  val     clean synthetic lines, held out from training.

If no real data could be fetched the golden set falls back to synthetic
lines, and the report says so rather than quietly pretending otherwise.
"""

import json
import random

from nastaliqops.config import Paths, ensure_dirs, load_params


def write_split(name, records):
    path = Paths.splits / (name + ".jsonl")
    handle = open(path, "w", encoding="utf-8")
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.close()
    print(name + ": " + str(len(records)) + " rows")


def main():
    params = load_params()
    ratios = params["data"]["splits"]
    seed = params["data"]["seed"]

    ensure_dirs()
    Paths.splits.mkdir(parents=True, exist_ok=True)

    handle = open(Paths.processed / "manifest.jsonl", "r", encoding="utf-8")
    lines = handle.readlines()
    handle.close()

    real_rows = []
    clean_rows = []
    harsh_rows = []

    for line in lines:
        line = line.strip()
        if line == "":
            continue
        record = json.loads(line)

        if record.get("is_real", False):
            real_rows.append(record)
            continue

        file_name = record.get("file_name", "")
        if file_name.startswith("stream_"):
            harsh_rows.append(record)
        else:
            clean_rows.append(record)

    random.seed(seed)
    random.shuffle(real_rows)
    random.shuffle(clean_rows)
    random.shuffle(harsh_rows)

    golden_rows = real_rows
    stream_rows = harsh_rows

    if len(golden_rows) == 0:
        print("no real rows available, falling back to synthetic for golden")
        take = int(len(clean_rows) * ratios["golden"])
        golden_rows = clean_rows[:take]
        clean_rows = clean_rows[take:]

    # whatever is left of the clean synthetic pool splits into train and val
    total_clean = len(clean_rows)
    train_share = ratios["train"] / (ratios["train"] + ratios["val"])
    train_end = int(total_clean * train_share)

    train_rows = []
    val_rows = []

    position = 0
    while position < total_clean:
        if position < train_end:
            train_rows.append(clean_rows[position])
        else:
            val_rows.append(clean_rows[position])
        position = position + 1

    write_split("train", train_rows)
    write_split("val", val_rows)
    write_split("golden", golden_rows)
    write_split("stream", stream_rows)

    summary = {}
    summary["train"] = len(train_rows)
    summary["val"] = len(val_rows)
    summary["golden"] = len(golden_rows)
    summary["stream"] = len(stream_rows)
    summary["golden_is_real"] = len(real_rows) > 0

    out_path = Paths.reports / "splits.json"
    handle = open(out_path, "w", encoding="utf-8")
    handle.write(json.dumps(summary, indent=2))
    handle.close()


if __name__ == "__main__":
    main()
