import json
import random
from pathlib import Path

from src.asr.whisper_engine import (
    WhisperEngine,
)
# from src.audio.microphone import (
#     record_audio,
# )
from src.audio.microphone import (
    record_until_silence,
)
from src.evaluation.metrics import (
    calculate_cer,
    calculate_wer,
)
from src.evaluation.performance import (
    calculate_rtf,
)


MANIFEST = Path(
    "datasets/metadata/golden_manifest.json"
)

# RECORD_SECONDS = 8


def main():

    with MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        samples = json.load(f)

    sample = random.choice(
        samples
    )

    expected = sample[
        "expected_transcript"
    ]

    print()
    print("=" * 72)
    print("PROMPTED LIVE ASR TEST")
    print("=" * 72)

    print()
    print(
        "Read the following sentence "
        "exactly as written:"
    )

    print()
    print(expected)

    print()
    input(
        "Press ENTER when ready..."
    )

    print()
    # print(
    #     f"Recording for "
    #     f"{RECORD_SECONDS} seconds..."
    # )

    # audio = record_audio(
    #     RECORD_SECONDS
    # )

    print()
    print(
        "Start speaking after this message."
    )
    print(
        "Recording will stop automatically "
        "after you finish."
    )

    audio = record_until_silence(
        trailing_silence_seconds=1.5,
        max_duration_seconds=35.0,
    )

    print(
        "Recording complete."
    )

    print()
    print(
        "Loading Whisper..."
    )

    engine = WhisperEngine(
        model_size="base",
        device="cpu",
        compute_type="int8",
    )

    # result = (
    #     engine.transcribe_array(
    #         audio.samples
    #     )
    # )
    result = engine.transcribe_array(
        audio.samples,
        use_vad=True,
    )

    wer = calculate_wer(
        expected,
        result.text,
    )

    cer = calculate_cer(
        expected,
        result.text,
    )

    rtf = calculate_rtf(
        result.inference_seconds,
        result.duration_seconds,
    )

    print()
    print("=" * 72)

    print(
        "Expected:"
    )
    print(expected)

    print()
    print(
        "Recognized:"
    )
    print(result.text)

    print()
    print("-" * 72)

    print(
        f"WER:            "
        f"{wer:.2%}"
    )

    print(
        f"CER:            "
        f"{cer:.2%}"
    )

    print(
        f"Inference:      "
        f"{result.inference_seconds:.2f}s"
    )

    print(
        f"Audio duration: "
        f"{result.duration_seconds:.2f}s"
    )

    print(
        f"RTF:            "
        f"{rtf:.2f}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()