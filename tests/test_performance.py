import pytest

from src.evaluation.performance import (
    calculate_rtf,
)


def test_rtf():

    assert calculate_rtf(
        inference_seconds=1.0,
        audio_duration_seconds=5.0,
    ) == pytest.approx(0.2)


def test_invalid_duration():

    with pytest.raises(ValueError):

        calculate_rtf(
            inference_seconds=1.0,
            audio_duration_seconds=0,
        )