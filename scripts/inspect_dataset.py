"""Print what a Hugging Face dataset repo actually contains.

load_dataset only works when the repo is laid out in a way it recognises
(parquet, csv, an imagefolder and so on). Some OCR datasets are published as
LMDB archives or zips instead, and then load_dataset reports the repo as
empty. This script shows the real file list so the ingest stage can be
pointed at the right thing.

    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py some-other/dataset
"""

import sys

from huggingface_hub import list_repo_files


DEFAULT_REPO = "abdur75648/UTRSet-Real"


def human(size):
    if size is None:
        return "?"
    value = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0:
            return str(round(value, 1)) + " " + unit
        value = value / 1024.0
    return str(round(value, 1)) + " TB"


def main():
    repo_id = DEFAULT_REPO
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]

    print("inspecting " + repo_id)
    print("")

    try:
        files = list_repo_files(repo_id, repo_type="dataset")
    except Exception as error:
        print("could not list the repo: " + str(error))
        sys.exit(1)

    print("total files: " + str(len(files)))
    print("")

    shown = 0
    for name in files:
        print("  " + name)
        shown = shown + 1
        if shown >= 60:
            print("  ... and " + str(len(files) - shown) + " more")
            break

    print("")
    print("extensions seen:")
    counts = {}
    for name in files:
        if "." in name:
            parts = name.split(".")
            extension = parts[len(parts) - 1].lower()
        else:
            extension = "(none)"
        if extension in counts:
            counts[extension] = counts[extension] + 1
        else:
            counts[extension] = 1

    for extension in counts:
        print("  ." + extension + ": " + str(counts[extension]))


if __name__ == "__main__":
    main()
