from nastaliqops.utils.urdu import is_mostly_urdu, normalise


def test_normalise_collapses_whitespace():
    assert normalise("  a   b  ") == "a b"


def test_normalise_maps_arabic_yeh_to_urdu_yeh():
    arabic_yeh = "ي"
    urdu_yeh = "ی"
    assert normalise(arabic_yeh) == urdu_yeh


def test_normalise_maps_arabic_kaf():
    arabic_kaf = "ك"
    keheh = "ک"
    assert normalise(arabic_kaf) == keheh


def test_normalise_strips_diacritics():
    with_fatha = "بَ"
    assert normalise(with_fatha) == "ب"


def test_normalise_handles_none():
    assert normalise(None) == ""


def test_is_mostly_urdu_true():
    assert is_mostly_urdu("اردو متن") is True


def test_is_mostly_urdu_false_for_english():
    assert is_mostly_urdu("plain english text") is False


def test_is_mostly_urdu_false_for_empty():
    assert is_mostly_urdu("") is False
