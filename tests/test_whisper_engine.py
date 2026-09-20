import pytest

from src.asr.whisper_engine import WhisperEngine


def test_missing_audio_file():
    engine = WhisperEngine(
        model_size="base",
        device="cpu",
        compute_type="int8",
    )

    with pytest.raises(FileNotFoundError):
        engine.transcribe("nonexistent.wav")