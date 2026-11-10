import pandas as pd
import pytest

pandera = pytest.importorskip("pandera")

from nastaliqops.data.schema import validate_frame


URDU_LINE = "اردو متن کی سطر"


def make_row(row_id, text=URDU_LINE, width=600, height=64):
    row = {}
    row["id"] = row_id
    row["file_name"] = "line_" + str(row_id).zfill(6) + ".png"
    row["text"] = text
    row["width"] = width
    row["height"] = height
    if height > 0:
        row["aspect_ratio"] = float(width) / float(height)
    else:
        row["aspect_ratio"] = 0.0
    return row


def test_clean_frame_has_no_failures():
    frame = pd.DataFrame([make_row(0), make_row(1), make_row(2)])
    failed, reasons = validate_frame(frame)
    assert failed == []
    assert reasons == {}


def test_english_text_is_rejected():
    frame = pd.DataFrame([make_row(0), make_row(1, text="plain english text")])
    failed, reasons = validate_frame(frame)
    assert 1 in failed
    assert 0 not in failed


def test_empty_text_is_rejected():
    frame = pd.DataFrame([make_row(0), make_row(1, text="")])
    failed, reasons = validate_frame(frame)
    assert 1 in failed


def test_tiny_image_is_rejected():
    frame = pd.DataFrame([make_row(0), make_row(1, width=4, height=2)])
    failed, reasons = validate_frame(frame)
    assert 1 in failed


def test_absurd_aspect_ratio_is_rejected():
    frame = pd.DataFrame([make_row(0), make_row(1, width=9000, height=20)])
    failed, reasons = validate_frame(frame)
    assert 1 in failed


def test_reasons_are_counted():
    frame = pd.DataFrame([make_row(0), make_row(1, text="english"), make_row(2, text="more english")])
    failed, reasons = validate_frame(frame)
    assert len(failed) == 2
    total = 0
    for key in reasons:
        total = total + reasons[key]
    assert total >= 2
