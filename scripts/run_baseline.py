import json
from pathlib import Path

from src.asr.whisper_engine import WhisperEngine
from src.evaluation.metrics import (
    calculate_cer,
    calculate_wer,
)


MANIFEST_PATH = Path(
    "datasets/metadata/golden_manifest.json"
)


def main():
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        samples = json.load(f)

    print("Loading Whisper model...")

    engine = WhisperEngine(
        model_size="base",
        device="cpu",
        compute_type="int8",
    )

    print()
    print(
        f"{'ID':<22}"
        f"{'WER':>8}"
        f"{'CER':>8}"
        f"{'Latency':>12}"
    )

    print("-" * 52)

    results = []

    for sample in samples:
        result = engine.transcribe(
            sample["audio_file"]
        )

        word_error_rate = calculate_wer(
            sample["expected_transcript"],
            result.text,
        )

        character_error_rate = calculate_cer(
            sample["expected_transcript"],
            result.text,
        )

        results.append(
            {
                "id": sample["id"],
                "expected": sample["expected_transcript"],
                "actual": result.text,
                "wer": word_error_rate,
                "cer": character_error_rate,
                "audio_duration_seconds": result.duration_seconds,
                "inference_seconds": result.inference_seconds,
            }
        )

        print(
            f"{sample['id']:<22}"
            f"{word_error_rate:>8.2%}"
            f"{character_error_rate:>8.2%}"
            f"{result.inference_seconds:>10.2f}s"
        )

    output = Path(
        "reports/baseline_results.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    average_wer = (
        sum(r["wer"] for r in results)
        / len(results)
    )

    average_cer = (
        sum(r["cer"] for r in results)
        / len(results)
    )

    average_latency = (
        sum(
            r["inference_seconds"]
            for r in results
        )
        / len(results)
    )

    print()
    print("=" * 52)

    print(
        f"Average WER:     {average_wer:.2%}"
    )

    print(
        f"Average CER:     {average_cer:.2%}"
    )

    print(
        f"Average latency: {average_latency:.2f}s"
    )

    print()
    print(
        f"Detailed results saved to {output}"
    )


if __name__ == "__main__":
    main()