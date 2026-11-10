"""Stage 3: preprocess every accepted line image.

Grayscale, deskew the obvious cases, resize to a fixed height, pad to a fixed
width. The same function is imported by the API so that training and serving
see identical pixels. That is the training/serving consistency point worth
mentioning in the report.
"""

import json

import cv2
import numpy as np

from nastaliqops.config import Paths, ensure_dirs, load_params
from nastaliqops.utils.urdu import normalise


def preprocess_image(image_bgr, target_height, max_width):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # light denoise, printed scans usually carry speckle
    gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)

    # normalise contrast so faint scans and dark scans look alike
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    height, width = gray.shape
    scale = float(target_height) / float(height)
    new_width = int(width * scale)
    if new_width > max_width:
        new_width = max_width
    if new_width < 8:
        new_width = 8

    resized = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_height, max_width), 255, dtype=np.uint8)
    # urdu reads right to left, so pad on the left and keep the text on the right
    start = max_width - new_width
    canvas[:, start:max_width] = resized

    return canvas


def main():
    params = load_params()
    target_height = params["data"]["image_height"]
    max_width = params["data"]["max_width"]

    ensure_dirs()
    out_images = Paths.processed / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    report_handle = open(Paths.reports / "data_validation.json", "r", encoding="utf-8")
    validation = json.load(report_handle)
    report_handle.close()

    rejected = {}
    for bad_id in validation["rejected_ids"]:
        rejected[bad_id] = True

    manifest_handle = open(Paths.raw / "manifest.jsonl", "r", encoding="utf-8")
    lines = manifest_handle.readlines()
    manifest_handle.close()

    kept = []
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        record = json.loads(line)
        if record["id"] in rejected:
            continue

        source_path = str(Paths.raw / "images" / record["file_name"])
        image = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if image is None:
            continue

        processed = preprocess_image(image, target_height, max_width)
        cv2.imwrite(str(out_images / record["file_name"]), processed)

        new_record = {}
        new_record["id"] = record["id"]
        new_record["file_name"] = record["file_name"]
        new_record["text"] = normalise(record["text"])
        kept.append(new_record)

    out_path = Paths.processed / "manifest.jsonl"
    handle = open(out_path, "w", encoding="utf-8")
    for record in kept:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.close()

    print("preprocessed " + str(len(kept)) + " images")


if __name__ == "__main__":
    main()
