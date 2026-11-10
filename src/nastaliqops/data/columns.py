"""Column detection for whichever OCR dataset gets pointed at the pipeline.

Kept in its own module with no heavy imports so that CI can unit test it
without installing torch and datasets.
"""


IMAGE_COLUMN_CANDIDATES = ["image", "img", "images", "image_path", "file", "pixel_values"]
TEXT_COLUMN_CANDIDATES = ["text", "label", "labels", "gt", "ground_truth", "transcription", "sentence", "caption"]


def find_column(available, candidates, what):
    """Return the first candidate present in the dataset's columns.

    Raises with the real column list rather than guessing, because a silent
    wrong guess here writes empty labels and the failure only shows up much
    later as a terrible CER.
    """
    for candidate in candidates:
        if candidate in available:
            print("using '" + candidate + "' as the " + what + " column")
            return candidate

    message = "could not find a " + what + " column. the dataset has: "
    message = message + ", ".join(available)
    raise RuntimeError(message)
