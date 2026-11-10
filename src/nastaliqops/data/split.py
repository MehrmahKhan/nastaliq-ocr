"""Stage 4: split into train, val, golden and stream.

golden is the frozen judge. No model ever trains on it, and promotion is
decided on it alone. stream is held back to act as live uploads later, so
drift monitoring and the human review queue have something to chew on
without needing real users on day one.
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

    records = []
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        records.append(json.loads(line))

    random.seed(seed)
    random.shuffle(records)

    total = len(records)
    train_end = int(total * ratios["train"])
    val_end = train_end + int(total * ratios["val"])
    golden_end = val_end + int(total * ratios["golden"])

    train_rows = []
    val_rows = []
    golden_rows = []
    stream_rows = []

    position = 0
    while position < total:
        record = records[position]
        if position < train_end:
            train_rows.append(record)
        elif position < val_end:
            val_rows.append(record)
        elif position < golden_end:
            golden_rows.append(record)
        else:
            stream_rows.append(record)
        position = position + 1

    write_split("train", train_rows)
    write_split("val", val_rows)
    write_split("golden", golden_rows)
    write_split("stream", stream_rows)


if __name__ == "__main__":
    main()
