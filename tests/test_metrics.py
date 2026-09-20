from src.evaluation.metrics import (
    calculate_cer,
    calculate_wer,
    normalize_text,
)


def test_normalization():
    text = "Hello, WORLD!"

    assert normalize_text(text) == "hello world"


def test_perfect_transcription_has_zero_wer():
    reference = "THE QUICK BROWN FOX"
    hypothesis = "The quick brown fox."

    assert calculate_wer(reference, hypothesis) == 0.0


def test_word_error_is_detected():
    reference = "the quick brown fox"
    hypothesis = "the quick red fox"

    score = calculate_wer(reference, hypothesis)

    assert score > 0.0


def test_character_error_is_detected():
    reference = "hello"
    hypothesis = "hallo"

    score = calculate_cer(reference, hypothesis)

    assert score > 0.0