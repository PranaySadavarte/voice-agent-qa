import re

from jiwer import cer, wer


def normalize_text(text: str) -> str:
    """
    Normalize ASR text before calculating error metrics.

    Whisper may add punctuation or change capitalization even
    when the recognized words are correct.
    """

    text = text.lower()

    # Remove punctuation except apostrophes.
    text = re.sub(r"[^\w\s']", " ", text)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def calculate_wer(reference: str, hypothesis: str) -> float:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    return wer(reference, hypothesis)


def calculate_cer(reference: str, hypothesis: str) -> float:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    return cer(reference, hypothesis)