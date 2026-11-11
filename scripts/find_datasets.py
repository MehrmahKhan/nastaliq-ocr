"""Search the Hugging Face Hub for Urdu OCR datasets that actually hold data.

Plenty of dataset repos on the Hub are stubs: a README pointing somewhere
else and nothing to load. This script searches for candidates, looks at the
real file list of each one, and reports which are usable, so the ingest
stage can be pointed at a dataset that exists rather than one that only has
a nice model card.

    python scripts/find_datasets.py
"""

import sys

from huggingface_hub import HfApi


QUERIES = [
    "urdu ocr",
    "urdu text recognition",
    "nastaliq",
    "urdu handwritten",
    "urdu",
]

# extensions that mean load_dataset has something to work with
LOADABLE = ["parquet", "arrow", "csv", "json", "jsonl", "tar", "zip"]
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]

MAX_PER_QUERY = 25
MAX_CHECKED = 45


def extensions_of(files):
    counts = {}
    for name in files:
        if "." in name:
            parts = name.split(".")
            extension = parts[len(parts) - 1].lower()
        else:
            extension = "none"
        if extension in counts:
            counts[extension] = counts[extension] + 1
        else:
            counts[extension] = 1
    return counts


def verdict_for(counts):
    loadable_files = 0
    for extension in LOADABLE:
        if extension in counts:
            loadable_files = loadable_files + counts[extension]

    image_files = 0
    for extension in IMAGE_EXTENSIONS:
        if extension in counts:
            image_files = image_files + counts[extension]

    if loadable_files > 0:
        return "USABLE", loadable_files, image_files
    if image_files > 20:
        return "IMAGES", loadable_files, image_files
    return "stub", loadable_files, image_files


def main():
    api = HfApi()

    seen = {}
    order = []

    for query in QUERIES:
        try:
            results = api.list_datasets(search=query, limit=MAX_PER_QUERY)
        except Exception as error:
            print("search failed for '" + query + "': " + str(error))
            continue

        for dataset in results:
            if dataset.id in seen:
                continue
            seen[dataset.id] = dataset
            order.append(dataset.id)

    print("found " + str(len(order)) + " candidate repos, checking file lists")
    print("")

    usable = []
    checked = 0

    for repo_id in order:
        if checked >= MAX_CHECKED:
            break
        checked = checked + 1

        try:
            files = api.list_repo_files(repo_id, repo_type="dataset")
        except Exception:
            continue

        counts = extensions_of(files)
        verdict, loadable_files, image_files = verdict_for(counts)

        if verdict == "stub":
            continue

        downloads = 0
        if hasattr(seen[repo_id], "downloads") and seen[repo_id].downloads is not None:
            downloads = seen[repo_id].downloads

        record = {}
        record["id"] = repo_id
        record["verdict"] = verdict
        record["files"] = len(files)
        record["loadable"] = loadable_files
        record["images"] = image_files
        record["downloads"] = downloads
        usable.append(record)

    if len(usable) == 0:
        print("nothing usable found, we will generate the dataset instead")
        sys.exit(0)

    # simple sort by downloads, highest first, plain loop on purpose
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            if usable[j]["downloads"] > usable[i]["downloads"]:
                swap = usable[i]
                usable[i] = usable[j]
                usable[j] = swap

    print("usable datasets:")
    print("")
    for record in usable:
        line = "  " + record["verdict"].ljust(7)
        line = line + record["id"].ljust(48)
        line = line + " files=" + str(record["files"]).ljust(6)
        line = line + " data=" + str(record["loadable"]).ljust(5)
        line = line + " imgs=" + str(record["images"]).ljust(6)
        line = line + " dl=" + str(record["downloads"])
        print(line)


if __name__ == "__main__":
    main()
