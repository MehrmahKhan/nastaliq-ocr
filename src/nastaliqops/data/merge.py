"""Stage 1c: merge the synthetic and real manifests into one.

Keeping them separate up to this point means either source can be
regenerated on its own, and it keeps the provenance of every row, which is
what the split stage uses to decide what the golden set is made of.
"""

import json

from nastaliqops.config import Paths, ensure_dirs


def read_manifest(path):
    if not path.exists():
        return []

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


def main():
    ensure_dirs()

    synthetic = read_manifest(Paths.raw / "synth" / "manifest.jsonl")
    real = read_manifest(Paths.raw / "real" / "manifest.jsonl")

    print("synthetic rows: " + str(len(synthetic)))
    print("real rows: " + str(len(real)))

    merged = []
    next_id = 0

    for record in synthetic:
        record["id"] = next_id
        if "source" not in record:
            record["source"] = "synthetic"
        record["is_real"] = False
        merged.append(record)
        next_id = next_id + 1

    for record in real:
        record["id"] = next_id
        record["is_real"] = True
        merged.append(record)
        next_id = next_id + 1

    out_path = Paths.raw / "manifest.jsonl"
    handle = open(out_path, "w", encoding="utf-8")
    for record in merged:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.close()

    print("merged " + str(len(merged)) + " rows into " + str(out_path))


if __name__ == "__main__":
    main()
