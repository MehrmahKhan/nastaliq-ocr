import random

import pytest

from nastaliqops.data.corpus import build_corpus, make_line, random_number
from nastaliqops.data.fonts import REQUIRED_CODEPOINTS, covers_urdu
from nastaliqops.utils.urdu import is_mostly_urdu


def test_make_line_is_urdu():
    rng = random.Random(0)
    for _ in range(20):
        line = make_line(rng)
        assert is_mostly_urdu(line)


def test_make_line_respects_word_count():
    rng = random.Random(1)
    line = make_line(rng, min_words=4, max_words=4)
    assert len(line.split()) == 4


def test_random_number_uses_urdu_digits():
    rng = random.Random(2)
    text = random_number(rng)
    assert len(text) >= 1
    for char in text:
        assert 0x06F0 <= ord(char) <= 0x06F9


def test_build_corpus_without_network_returns_enough_lines():
    lines = build_corpus(50, seed=7, repo_id="")
    assert len(lines) == 50
    for line in lines:
        assert len(line.strip()) > 0


def test_build_corpus_is_deterministic():
    first = build_corpus(30, seed=11, repo_id="")
    second = build_corpus(30, seed=11, repo_id="")
    assert first == second


def test_required_codepoints_are_urdu_letters():
    for codepoint in REQUIRED_CODEPOINTS:
        assert 0x0600 <= codepoint <= 0x06FF


def test_covers_urdu_rejects_a_latin_font(tmp_path):
    fonttools = pytest.importorskip("fontTools")

    import glob
    candidates = glob.glob("/mnt/skills/examples/canvas-design/canvas-fonts/*.ttf")
    if len(candidates) == 0:
        pytest.skip("no latin font available to test against")

    from pathlib import Path
    assert covers_urdu(Path(candidates[0])) is False


def test_covers_urdu_rejects_a_missing_file(tmp_path):
    from pathlib import Path
    assert covers_urdu(Path(tmp_path) / "nope.ttf") is False
