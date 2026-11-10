"""Stage 1: pull a subset of UTRSet-Real and write it to data/raw.

UTRSet-Real is CC BY-NC 4.0 (research and academic use), so it is fine for a
course project as long as the dataset is cited. The raw folder is tracked by
DVC, not by git.
"""

import json
import random

from datasets import load_dataset
from PIL import Image

from nastaliqops.config import Paths, ensure_dirs, load_params
from nastaliqops.data.columns import (
    IMAGE_COLUMN_CANDIDATES,
    TEXT_COLUMN_CANDIDATES,
    find_column,
)


def main():
    params = load_params()
    data_params = params["data"]

    ensure_dirs()
    images_dir = Paths.raw / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("loading " + data_params["hf_dataset"] + " from the hub")
    dataset = load_dataset(data_params["hf_dataset"], split="train")

    available = list(dataset.column_names)
    print("columns found: " + ", ".join(available))

    image_column = find_column(available, IMAGE_COLUMN_CANDIDATES, "image")
    text_column = find_column(available, TEXT_COLUMN_CANDIDATES, "text")

    indices = list(range(len(dataset)))
    random.seed(data_params["seed"])
    random.shuffle(indices)

    wanted = data_params["subset_size"]
    if wanted > len(indices):
        wanted = len(indices)

    manifest = []
    position = 0
    while position < wanted:
        row_index = indices[position]
        row = dataset[row_index]

        image = row[image_column]

        # some datasets store a path string instead of a decoded image
        if isinstance(image, str):
            image = Image.open(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        file_name = "line_" + str(position).zfill(6) + ".png"
        image.save(images_dir / file_name)

        record = {}
        record["id"] = position
        record["file_name"] = file_name
        record["text"] = row[text_column]
        record["source"] = data_params["hf_dataset"]
        manifest.append(record)

        position = position + 1

    out_path = Paths.raw / "manifest.jsonl"
    handle = open(out_path, "w", encoding="utf-8")
    for record in manifest:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.close()

    print("wrote " + str(len(manifest)) + " lines to " + str(out_path))


if __name__ == "__main__":
    main()
