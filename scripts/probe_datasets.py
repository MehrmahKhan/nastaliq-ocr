"""Look at one row of each candidate dataset without downloading it.

streaming=True makes the Hub serve rows one at a time, so this costs a few
hundred kilobytes instead of several gigabytes. For each candidate it
prints the split names, the column names, and what the first row actually
holds, which is what the ingest stage needs to know.

    python scripts/probe_datasets.py
"""

from datasets import load_dataset


CANDIDATES = [
    "PuristanLabs1/urdu-ocr-1M",
    "deepcopy/MMU-OCR-21-Urdu-TextLines",
    "Ali000007/Urdu-Ocr-Dataset",
    "Ali000007/urdu-shayari-ocr-lines",
    "Khurram123/urdu-poetry-nastaleeq-ocr",
    "oddadmix/qaari-0.1-ocr-urdu-news-dataset-small",
    "vettedlance/urdu-nastaliq-1908-eval-sample",
    "ebadhussain20/urdu_ocr",
    "tfxhk/urdu_ocr",
]


def describe_value(value):
    kind = type(value).__name__

    if hasattr(value, "size") and hasattr(value, "mode"):
        return "image " + str(value.size) + " " + str(value.mode)

    if isinstance(value, str):
        shown = value
        if len(shown) > 60:
            shown = shown[:60] + "..."
        return "str(" + str(len(value)) + ") " + repr(shown)

    if isinstance(value, dict):
        return "dict keys=" + str(list(value.keys()))

    if isinstance(value, bytes):
        return "bytes(" + str(len(value)) + ")"

    return kind + " " + repr(value)[:60]


def probe(repo_id):
    print("")
    print("=" * 70)
    print(repo_id)
    print("=" * 70)

    try:
        streamed = load_dataset(repo_id, streaming=True)
    except Exception as error:
        message = str(error)
        if len(message) > 200:
            message = message[:200] + "..."
        print("  could not open: " + message)
        return

    split_names = list(streamed.keys())
    print("  splits: " + ", ".join(split_names))

    first_split = split_names[0]
    stream = streamed[first_split]

    try:
        iterator = iter(stream)
        row = next(iterator)
    except Exception as error:
        message = str(error)
        if len(message) > 200:
            message = message[:200] + "..."
        print("  could not read a row: " + message)
        return

    print("  columns in '" + first_split + "':")
    for key in row:
        print("    " + key.ljust(20) + " -> " + describe_value(row[key]))


def main():
    for repo_id in CANDIDATES:
        probe(repo_id)

    print("")
    print("done")


if __name__ == "__main__":
    main()
