from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from faster_whisper import WhisperModel


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration_seconds: float
    inference_seconds: float


class WhisperEngine:
    """Wrapper around Whisper for repeatable ASR evaluation."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(
        self,
        audio_path: str | Path,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file from disk.
        """

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        start = perf_counter()

        segments, info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
        )

        segments = list(segments)

        inference_seconds = (
            perf_counter() - start
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        return TranscriptionResult(
            text=text,
            language=info.language,
            duration_seconds=info.duration,
            inference_seconds=inference_seconds,
        )

    def transcribe_array(
        self,
        audio: np.ndarray,
        use_vad: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe a mono 16 kHz NumPy waveform.

        use_vad=True enables faster-whisper's VAD filtering,
        which is useful for live microphone input.
        """

        if audio.ndim != 1:
            raise ValueError(
                "Audio must be a mono 1D waveform."
            )

        if len(audio) == 0:
            raise ValueError(
                "Audio waveform is empty."
            )

        audio = audio.astype(
            np.float32
        )

        transcription_options = {
            "beam_size": 5,
            "language": "en",
        }

        if use_vad:
            transcription_options[
                "vad_filter"
            ] = True

            transcription_options[
                "vad_parameters"
            ] = {
                "min_silence_duration_ms": 500
            }

        start = perf_counter()

        segments, info = self.model.transcribe(
            audio,
            **transcription_options,
        )

        segments = list(segments)

        inference_seconds = (
            perf_counter() - start
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        duration_seconds = (
            len(audio) / 16000
        )

        return TranscriptionResult(
            text=text,
            language=info.language,
            duration_seconds=duration_seconds,
            inference_seconds=inference_seconds,
        )