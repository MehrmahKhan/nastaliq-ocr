"""Stage 1a: render synthetic Urdu line images.

This is the training data generator. Every image is produced with a known
label, so there is no annotation error anywhere in the training set, and
the amount of data is limited only by how long you let it run.

The variation applied here is the whole point. A generator that renders
clean black text on white paper at one size teaches the model nothing about
real scans, so each line gets a randomly chosen font, size, padding,
background tint, ink darkness, blur, noise and slight rotation. The
severity of that variation is a parameter, which is what lets the pipeline
produce a clean training set and a deliberately harsher stream set for the
drift monitor to detect.
"""

import json
import random
import sys

import numpy as np

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from nastaliqops.config import Paths, ensure_dirs, load_params
from nastaliqops.data.corpus import build_corpus
from nastaliqops.data.fonts import resolve_fonts
from nastaliqops.utils.urdu import normalise


def load_font_objects(font_paths, sizes):
    """Pre-load every font at every size, since loading is slow."""
    loaded = []
    for font_path in font_paths:
        for size in sizes:
            try:
                font = ImageFont.truetype(str(font_path), size)
                loaded.append(font)
            except Exception as error:
                print("  could not load " + str(font_path) + " at " + str(size) + ": " + str(error))

    if len(loaded) == 0:
        print("no font could be loaded, cannot generate images")
        sys.exit(1)

    return loaded


def render_line(text, font, severity, rng):
    """Draw one line of Urdu and return it as a grayscale numpy array."""

    # measure first on a scratch image
    scratch = Image.new("L", (10, 10), 255)
    drawer = ImageDraw.Draw(scratch)

    try:
        box = drawer.textbbox((0, 0), text, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        offset_x = box[0]
        offset_y = box[1]
    except Exception:
        text_width = len(text) * 20
        text_height = 40
        offset_x = 0
        offset_y = 0

    if text_width < 10:
        text_width = 10
    if text_height < 10:
        text_height = 10

    pad_x = rng.randint(10, 30)
    pad_y = rng.randint(8, 22)

    width = text_width + pad_x * 2
    height = text_height + pad_y * 2

    # paper is rarely pure white
    paper = rng.randint(238, 255)
    if severity > 0.5:
        paper = rng.randint(215, 252)

    canvas = Image.new("L", (width, height), paper)
    drawer = ImageDraw.Draw(canvas)

    # ink is rarely pure black
    ink = rng.randint(0, 60)
    if severity > 0.5:
        ink = rng.randint(20, 110)

    drawer.text((pad_x - offset_x, pad_y - offset_y), text, font=font, fill=ink)

    # slight rotation, scans are never perfectly straight
    max_angle = 0.6 + severity * 2.0
    angle = rng.uniform(-max_angle, max_angle)
    canvas = canvas.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=paper)

    # blur
    if rng.random() < (0.25 + severity * 0.5):
        radius = rng.uniform(0.3, 0.6 + severity * 1.6)
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius))

    array = np.array(canvas).astype(np.float32)

    # sensor noise
    noise_amount = 2.0 + severity * 18.0
    noise = np.random.normal(0.0, noise_amount, array.shape)
    array = array + noise

    # uneven lighting across the page
    if rng.random() < (0.2 + severity * 0.5):
        gradient_strength = rng.uniform(5.0, 12.0 + severity * 30.0)
        columns = np.linspace(-1.0, 1.0, array.shape[1])
        gradient = np.tile(columns, (array.shape[0], 1)) * gradient_strength
        array = array + gradient

    array = np.clip(array, 0, 255)
    return array.astype(np.uint8)


def generate(count, severity, seed, font_objects, corpus, start_id, images_dir, prefix):
    rng = random.Random(seed)
    np.random.seed(seed)

    records = []
    position = 0

    while position < count:
        text = corpus[position % len(corpus)]
        font = rng.choice(font_objects)

        try:
            array = render_line(text, font, severity, rng)
        except Exception as error:
            print("  render failed: " + str(error))
            position = position + 1
            continue

        file_name = prefix + "_" + str(start_id + position).zfill(6) + ".png"
        Image.fromarray(array).save(images_dir / file_name)

        record = {}
        record["id"] = start_id + position
        record["file_name"] = file_name
        record["rel_path"] = "synth/images/" + file_name
        record["text"] = normalise(text)
        record["source"] = "synthetic"
        record["severity"] = severity
        records.append(record)

        position = position + 1

        if position % 250 == 0:
            print("  " + str(position) + " / " + str(count))

    return records


def main():
    params = load_params()
    synth_params = params["synth"]

    ensure_dirs()
    images_dir = Paths.raw / "synth" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("resolving fonts")
    font_paths = resolve_fonts()
    font_objects = load_font_objects(font_paths, synth_params["font_sizes"])
    print("loaded " + str(len(font_objects)) + " font/size combinations")

    clean_count = synth_params["train_count"]
    harsh_count = synth_params["stream_count"]

    print("building the text corpus")
    corpus = build_corpus(
        clean_count + harsh_count,
        seed=params["data"]["seed"],
        repo_id=synth_params.get("text_corpus_repo", ""),
        text_column=synth_params.get("text_corpus_column", "text"),
    )
    print("corpus ready: " + str(len(corpus)) + " lines")

    all_records = []

    print("rendering " + str(clean_count) + " clean lines for training")
    clean_records = generate(
        clean_count,
        synth_params["clean_severity"],
        params["data"]["seed"],
        font_objects,
        corpus,
        0,
        images_dir,
        "synth",
    )
    for record in clean_records:
        all_records.append(record)

    print("rendering " + str(harsh_count) + " degraded lines to act as live traffic")
    harsh_records = generate(
        harsh_count,
        synth_params["harsh_severity"],
        params["data"]["seed"] + 1,
        font_objects,
        corpus[clean_count:],
        clean_count,
        images_dir,
        "stream",
    )
    for record in harsh_records:
        all_records.append(record)

    out_path = Paths.raw / "synth" / "manifest.jsonl"
    handle = open(out_path, "w", encoding="utf-8")
    for record in all_records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.close()

    print("wrote " + str(len(all_records)) + " synthetic lines to " + str(out_path))


if __name__ == "__main__":
    main()
