from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000
CHANNELS = 1

import time


def record_until_silence(
    sample_rate: int = SAMPLE_RATE,
    block_seconds: float = 0.1,
    silence_threshold: float = 0.008,
    trailing_silence_seconds: float = 1.5,
    start_timeout_seconds: float = 5.0,
    max_duration_seconds: float = 35.0,
) -> RecordedAudio:
    """
    Record microphone input until the speaker stops talking.

    Recording begins immediately, waits for speech to cross the
    RMS threshold, and stops after sustained trailing silence.
    """

    block_size = int(
        sample_rate * block_seconds
    )

    frames = []

    speech_started = False
    silence_duration = 0.0
    total_duration = 0.0

    start_time = time.monotonic()

    print("Listening...")

    with sd.InputStream(
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
        blocksize=block_size,
    ) as stream:

        while True:

            audio_block, overflowed = stream.read(
                block_size
            )

            if overflowed:
                print(
                    "[Audio warning] input overflow"
                )

            block = (
                audio_block[:, 0]
                .copy()
                .astype(np.float32)
            )

            frames.append(block)

            total_duration += block_seconds

            rms = float(
                np.sqrt(
                    np.mean(
                        block.astype(
                            np.float64
                        ) ** 2
                    )
                )
            )

            if rms >= silence_threshold:

                if not speech_started:
                    print("Speech detected.")

                speech_started = True
                silence_duration = 0.0

            elif speech_started:

                silence_duration += (
                    block_seconds
                )

            if (
                speech_started
                and silence_duration
                >= trailing_silence_seconds
            ):
                break

            if (
                not speech_started
                and time.monotonic()
                - start_time
                >= start_timeout_seconds
            ):
                raise TimeoutError(
                    "No speech detected."
                )

            if (
                total_duration
                >= max_duration_seconds
            ):
                print(
                    "Maximum recording duration reached."
                )
                break

    audio = np.concatenate(frames)

    return RecordedAudio(
        samples=audio,
        sample_rate=sample_rate,
        duration_seconds=(
            len(audio) / sample_rate
        ),
    )


@dataclass
class RecordedAudio:
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float


def record_audio(
    duration_seconds: float,
    sample_rate: int = SAMPLE_RATE,
) -> RecordedAudio:
    """
    Record mono microphone audio as float32.

    The returned waveform is compatible with faster-whisper
    when recorded at 16 kHz.
    """

    frames = int(
        duration_seconds * sample_rate
    )

    audio = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
        blocking=True,
    )

    # Convert (N, 1) -> (N,)
    audio = np.squeeze(audio)

    return RecordedAudio(
        samples=audio.astype(np.float32),
        sample_rate=sample_rate,
        duration_seconds=len(audio)
        / sample_rate,
    )


class MicrophoneStream:
    """
    Continuous microphone capture using sounddevice InputStream.

    Audio blocks are placed into a thread-safe queue.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        block_seconds: float = 0.5,
    ):
        self.sample_rate = sample_rate

        self.block_size = int(
            block_seconds * sample_rate
        )

        self.queue: queue.Queue[
            np.ndarray
        ] = queue.Queue()

        self.stream = None

    def _callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ):
        if status:
            print(
                f"[Audio warning] {status}"
            )

        self.queue.put(
            indata[:, 0]
            .copy()
            .astype(np.float32)
        )

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback,
        )

        self.stream.start()

    def read_chunk(
        self,
        duration_seconds: float,
    ) -> np.ndarray:

        target_samples = int(
            duration_seconds
            * self.sample_rate
        )

        blocks = []
        collected = 0

        while collected < target_samples:

            block = self.queue.get()

            blocks.append(block)

            collected += len(block)

        audio = np.concatenate(blocks)

        return audio[
            :target_samples
        ].astype(np.float32)

    def stop(self):

        if self.stream is not None:

            self.stream.stop()
            self.stream.close()

            self.stream = None