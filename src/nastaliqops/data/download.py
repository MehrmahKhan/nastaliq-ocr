"""Stage 1b: stream real Urdu line images for the golden evaluation set.

Only a few hundred rows are taken, and they are taken in streaming mode,
so the Hub serves them one at a time over HTTP and nothing is cached. That
matters: urdu-ocr-1M is a million samples, and downloading it in full to
keep 400 rows would be absurd on a laptop.

These real images are what the frozen golden set is built from. The model
trains on synthetic data and is judged on real data, which is the honest
way round and is what the UTRNet paper did too.

Several candidate datasets are tried in order, because dataset repos on the
Hub disappear, get renamed, or turn out to hold no data at all.
"""

import json
import sys

from PIL import Image

from nastaliqops.config import Paths, ensure_dirs, load_params
from nastaliqops.data.columns import (
    IMAGE_COLUMN_CANDIDATES,
    TEXT_COLUMN_CANDIDATES,
    find_column,
)
from nastaliqops.utils.urdu import is_mostly_urdu, normalise


def open_stream(repo_id):
    """Return an iterable over the dataset's first split, or None."""
    from datasets import load_dataset

    streamed = load_dataset(repo_id, streaming=True)
    split_names = list(streamed.keys())
    if len(split_names) == 0:
        return None

    preferred = None
    for candidate in ["train", "test", "validation"]:
        if candidate in split_names:
            preferred = candidate
            break
    if preferred is None:
        preferred = split_names[0]

    print("  split: " + preferred + " (of " + ", ".join(split_names) + ")")
    return streamed[preferred]


def coerce_image(value):
    """Rows store images as PIL objects, dicts with bytes, or paths."""
    if value is None:
        return None

    if hasattr(value, "size") and hasattr(value, "mode"):
        return value

    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            import io
            return Image.open(io.BytesIO(value["bytes"]))
        if "path" in value and value["path"] is not None:
            return Image.open(value["path"])
        return None

    if isinstance(value, str):
        try:
            return Image.open(value)
        except Exception:
            return None

    return None


def harvest(repo_id, wanted, images_dir, start_id):
    """Pull up to `wanted` usable rows from one dataset. Returns records."""
    print("")
    print("trying " + repo_id)

    try:
        stream = open_stream(repo_id)
    except Exception as error:
        message = str(error)
        if len(message) > 160:
            message = message[:160] + "..."
        print("  could not open: " + message)
        return []

    if stream is None:
        print("  no splits")
        return []

    iterator = iter(stream)

    try:
        first_row = next(iterator)
    except Exception as error:
        print("  could not read a row: " + str(error))
        return []

    available = list(first_row.keys())
    print("  columns: " + ", ".join(available))

    try:
        image_column = find_column(available, IMAGE_COLUMN_CANDIDATES, "image")
        text_column = find_column(available, TEXT_COLUMN_CANDIDATES, "text")
    except RuntimeError as error:
        print("  " + str(error))
        return []

    records = []
    rejected = 0
    row = first_row

    while True:
        if len(records) >= wanted:
            break

        try:
            text = row.get(text_column)
            image = coerce_image(row.get(image_column))

            if image is None or not isinstance(text, str):
                rejected = rejected + 1
            else:
                cleaned = normalise(text)

                if len(cleaned) < 2 or not is_mostly_urdu(cleaned):
                    rejected = rejected + 1
                else:
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    width, height = image.size
                    if width < 16 or height < 8:
                        rejected = rejected + 1
                    else:
                        index = start_id + len(records)
                        file_name = "real_" + str(index).zfill(6) + ".png"
                        image.save(images_dir / file_name)

                        record = {}
                        record["id"] = index
                        record["file_name"] = file_name
                        record["rel_path"] = "real/images/" + file_name
                        record["text"] = cleaned
                        record["source"] = repo_id
                        records.append(record)

                        if len(records) % 100 == 0:
                            print("  " + str(len(records)) + " / " + str(wanted))
        except Exception:
            rejected = rejected + 1

        if rejected > wanted * 4 and len(records) == 0:
            print("  nothing usable in the first rows, moving on")
            return []

        try:
            row = next(iterator)
        except StopIteration:
            break
        except Exception:
            break

    print("  kept " + str(len(records)) + ", skipped " + str(rejected))
    return records


def main():
    params = load_params()
    data_params = params["data"]

    ensure_dirs()
    images_dir = Paths.raw / "real" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    wanted = data_params["real_count"]
    candidates = data_params["hf_candidates"]

    collected = []

    for repo_id in candidates:
        remaining = wanted - len(collected)
        if remaining <= 0:
            break

        records = harvest(repo_id, remaining, images_dir, len(collected))
        for record in records:
            collected.append(record)

    print("")
    if len(collected) == 0:
        print("no real data could be fetched from any candidate.")
        print("the pipeline can still run fully synthetic, but the golden set")
        print("will not be real-world. set data.allow_synthetic_golden to true")
        print("in params.yaml if you want to proceed anyway.")
        if not data_params.get("allow_synthetic_golden", False):
            sys.exit(1)

    out_path = Paths.raw / "real" / "manifest.jsonl"
    handle = open(out_path, "w", encoding="utf-8")
    for record in collected:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.close()

    print("wrote " + str(len(collected)) + " real lines to " + str(out_path))


if __name__ == "__main__":
    main()
