import pytest

from nastaliqops.data.columns import (
    IMAGE_COLUMN_CANDIDATES,
    TEXT_COLUMN_CANDIDATES,
    find_column,
)


def test_finds_the_obvious_names():
    available = ["image", "text"]
    assert find_column(available, IMAGE_COLUMN_CANDIDATES, "image") == "image"
    assert find_column(available, TEXT_COLUMN_CANDIDATES, "text") == "text"


def test_finds_an_alternative_text_name():
    available = ["image", "label"]
    assert find_column(available, TEXT_COLUMN_CANDIDATES, "text") == "label"


def test_finds_an_alternative_image_name():
    available = ["img", "transcription"]
    assert find_column(available, IMAGE_COLUMN_CANDIDATES, "image") == "img"
    assert find_column(available, TEXT_COLUMN_CANDIDATES, "text") == "transcription"


def test_prefers_the_first_candidate_when_several_match():
    available = ["label", "text", "image"]
    assert find_column(available, TEXT_COLUMN_CANDIDATES, "text") == "text"


def test_raises_and_names_the_real_columns():
    available = ["foo", "bar"]
    with pytest.raises(RuntimeError) as error:
        find_column(available, TEXT_COLUMN_CANDIDATES, "text")
    assert "foo" in str(error.value)
    assert "bar" in str(error.value)
