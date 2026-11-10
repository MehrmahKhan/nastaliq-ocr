import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from nastaliqops.monitoring.features import FEATURE_NAMES, image_features
from nastaliqops.monitoring.drift import fallback_report, fallback_html

import pandas as pd


def make_image(value, size=(64, 256)):
    return np.full(size, value, dtype=np.uint8)


def make_text_like(brightness=255, ink=40):
    image = np.full((64, 256), brightness, dtype=np.uint8)
    image[20:44, 30:220] = ink
    return image


def test_features_have_every_name():
    features = image_features(make_text_like())
    for name in FEATURE_NAMES:
        assert name in features


def test_mean_intensity_tracks_brightness():
    bright = image_features(make_image(220))
    dark = image_features(make_image(40))
    assert bright["mean_intensity"] > dark["mean_intensity"]


def test_ink_ratio_is_higher_with_more_ink():
    thin = image_features(make_text_like(ink=40))
    thick = np.full((64, 256), 255, dtype=np.uint8)
    thick[5:59, 10:246] = 40
    thick_features = image_features(thick)
    assert thick_features["ink_ratio"] > thin["ink_ratio"]


def test_aspect_ratio_is_width_over_height():
    features = image_features(make_image(128, size=(50, 200)))
    assert abs(features["aspect_ratio"] - 4.0) < 1e-9


def build_frame(brightness_values):
    rows = []
    for value in brightness_values:
        rows.append(image_features(make_text_like(brightness=value)))
    return pd.DataFrame(rows)


def test_fallback_reports_no_drift_on_similar_data():
    reference = build_frame([250, 248, 252, 249, 251, 250, 247, 253])
    current = build_frame([250, 249, 251, 250, 248, 252, 250, 249])
    summary = fallback_report(reference, current)
    assert summary["columns_drifted"] == 0


def test_fallback_detects_a_brightness_shift():
    reference = build_frame([250, 248, 252, 249, 251, 250, 247, 253])
    current = build_frame([120, 118, 122, 119, 121, 120, 117, 123])
    summary = fallback_report(reference, current)
    assert summary["columns_drifted"] > 0
    assert summary["drift_share"] > 0.0


def test_fallback_html_mentions_every_feature_it_scored():
    reference = build_frame([250, 248, 252, 249, 251, 250, 247, 253])
    current = build_frame([120, 118, 122, 119, 121, 120, 117, 123])
    summary = fallback_report(reference, current)
    summary["detector"] = "fallback"
    html = fallback_html(summary)
    assert "Drift report" in html
    for name in summary["by_column"]:
        assert name in html
