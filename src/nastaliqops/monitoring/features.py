"""Turn a line image into a handful of numbers that drift can be measured on.

You cannot run a statistical drift test on raw pixels in any meaningful way,
so each image is reduced to simple, interpretable properties. If someone
starts uploading darker phone photos instead of clean scans, mean_intensity
and blur_score move, and that is exactly the signal worth alerting on.
"""

import cv2
import numpy as np


def image_features(gray):
    """Return a dict of scalar features for one grayscale image."""
    features = {}

    height, width = gray.shape
    features["width"] = float(width)
    features["height"] = float(height)
    if height > 0:
        features["aspect_ratio"] = float(width) / float(height)
    else:
        features["aspect_ratio"] = 0.0

    features["mean_intensity"] = float(np.mean(gray))
    features["std_intensity"] = float(np.std(gray))

    # variance of the laplacian is the usual cheap sharpness proxy, a low
    # value means a blurry or out of focus image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features["blur_score"] = float(np.var(laplacian))

    # how much of the image is ink rather than paper, after otsu
    threshold_value, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_pixels = 0
    total_pixels = binary.size
    dark_pixels = int(np.count_nonzero(binary == 0))
    if total_pixels > 0:
        features["ink_ratio"] = float(dark_pixels) / float(total_pixels)
    else:
        features["ink_ratio"] = 0.0

    features["otsu_threshold"] = float(threshold_value)
    return features


def features_for_file(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return image_features(gray)


FEATURE_NAMES = [
    "width",
    "height",
    "aspect_ratio",
    "mean_intensity",
    "std_intensity",
    "blur_score",
    "ink_ratio",
    "otsu_threshold",
]
