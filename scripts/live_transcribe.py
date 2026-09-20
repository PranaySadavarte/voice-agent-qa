from datetime import datetime

import numpy as np

from src.asr.whisper_engine import (
    WhisperEngine,
)
from src.audio.microphone import (
    MicrophoneStream,
)
from src.evaluation.performance import (
    calculate_rtf,
)


CHUNK_SECONDS = 4.0


def audio_rms(
    audio: np.ndarray,
) -> float:

    if len(audio) == 0:
        return 0.0

    return float(
        np.sqrt(
            np.mean(
                audio.astype(
                    np.float64
                ) ** 2
            )
        )
    )


def main():

    print()
    print("=" * 72)
    print("VOICE AI LIVE TRANSCRIPTION")
    print("=" * 72)

    print()
    print(
        "Loading Whisper model..."
    )

    engine = WhisperEngine(
        model_size="base",
        device="cpu",
        compute_type="int8",
    )

    microphone = MicrophoneStream()

    print(
        "Microphone ready."
    )

    print(
        "Speak normally. "
        "Press Ctrl+C to stop."
    )

    print()

    microphone.start()

    try:

        while True:

            audio = (
                microphone.read_chunk(
                    CHUNK_SECONDS
                )
            )

            # Ignore near-silent chunks.
            rms = audio_rms(audio)

            if rms < 0.002:

                print(
                    "[silence]"
                )

                continue

            # result = (
            #     engine.transcribe_array(
            #         audio
            #     )
            # )

            result = (
            engine.transcribe_array(
                audio,
                use_vad=True,
                )
            )

            if not result.text:

                print(
                    "[speech not recognized]"
                )

                continue

            rtf = calculate_rtf(
                result.inference_seconds,
                result.duration_seconds,
            )

            timestamp = (
                datetime.now()
                .strftime("%H:%M:%S")
            )

            print(
                f"[{timestamp}] "
                f"{result.text}"
            )

            print(
                f"    latency="
                f"{result.inference_seconds:.2f}s "
                f"| RTF={rtf:.2f}"
            )

    except KeyboardInterrupt:

        print()
        print(
            "Stopping live transcription..."
        )

    finally:

        microphone.stop()


if __name__ == "__main__":
    main()