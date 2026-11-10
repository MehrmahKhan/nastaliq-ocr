"""Reading the split files.

Deliberately free of torch and any model library, so that the data stages,
the drift monitor and CI can all read a split on a laptop that has no
model stack installed.
"""

import json

from nastaliqops.config import Paths


def read_split(name):
    path = Paths.splits / (name + ".jsonl")
    handle = open(path, "r", encoding="utf-8")
    lines = handle.readlines()
    handle.close()

    records = []
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        records.append(json.loads(line))
    return records
