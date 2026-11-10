from nastaliqops.utils.metrics import (
    character_error_rate,
    edit_distance,
    exact_match_rate,
    word_error_rate,
)


def test_edit_distance_identical():
    assert edit_distance("abc", "abc") == 0


def test_edit_distance_substitution():
    assert edit_distance("abc", "abd") == 1


def test_edit_distance_insertion():
    assert edit_distance("abc", "abcd") == 1


def test_cer_perfect():
    references = ["hello", "world"]
    hypotheses = ["hello", "world"]
    assert character_error_rate(references, hypotheses) == 0.0


def test_cer_one_wrong_character():
    references = ["abcde"]
    hypotheses = ["abcdx"]
    assert abs(character_error_rate(references, hypotheses) - 0.2) < 1e-9


def test_wer_counts_words():
    references = ["the quick brown fox"]
    hypotheses = ["the quick brown dog"]
    assert abs(word_error_rate(references, hypotheses) - 0.25) < 1e-9


def test_exact_match():
    references = ["a", "b", "c"]
    hypotheses = ["a", "x", "c"]
    assert abs(exact_match_rate(references, hypotheses) - (2.0 / 3.0)) < 1e-9
