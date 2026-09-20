import csv
import json
from collections import defaultdict
from pathlib import Path

from src.asr.whisper_engine import WhisperEngine
from src.evaluation.metrics import (
    calculate_cer,
    calculate_wer,
)


NOISE_MANIFEST = Path(
    "datasets/metadata/noise_manifest.json"
)

BASELINE_RESULTS = Path(
    "reports/baseline_results.json"
)

RESULTS_PATH = Path(
    "reports/noise_results.json"
)

SUMMARY_PATH = Path(
    "reports/noise_summary.csv"
)


def mean(values):
    return (
        sum(values) / len(values)
        if values
        else 0.0
    )


def main():
    with NOISE_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        samples = json.load(f)

    with BASELINE_RESULTS.open(
        "r",
        encoding="utf-8",
    ) as f:
        baseline_results = json.load(f)

    baseline_wer = mean(
        [
            result["wer"]
            for result in baseline_results
        ]
    )

    baseline_cer = mean(
        [
            result["cer"]
            for result in baseline_results
        ]
    )

    baseline_latency = mean(
        [
            result["inference_seconds"]
            for result in baseline_results
        ]
    )

    print("Loading Whisper model...")

    engine = WhisperEngine(
        model_size="base",
        device="cpu",
        compute_type="int8",
    )

    results = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
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

        record = {
            **sample,
            "actual_transcript": result.text,
            "wer": word_error_rate,
            "cer": character_error_rate,
            "inference_seconds": (
                result.inference_seconds
            ),
            "audio_duration_seconds": (
                result.duration_seconds
            ),
        }

        results.append(record)

        print(
            f"[{index:02}/{len(samples)}] "
            f"{sample['scenario']:<10} "
            f"WER={word_error_rate:>7.2%} "
            f"latency="
            f"{result.inference_seconds:.2f}s"
        )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    groups = defaultdict(list)

    for result in results:
        groups[
            result["scenario"]
        ].append(result)

    summary_rows = []

    # Clean baseline first.
    summary_rows.append(
        {
            "scenario": "clean",
            "snr_db": "",
            "samples": len(
                baseline_results
            ),
            "average_wer": baseline_wer,
            "average_cer": baseline_cer,
            "average_latency": (
                baseline_latency
            ),
            "wer_delta": 0.0,
        }
    )

    for scenario, scenario_results in groups.items():

        avg_wer = mean(
            [
                result["wer"]
                for result
                in scenario_results
            ]
        )

        avg_cer = mean(
            [
                result["cer"]
                for result
                in scenario_results
            ]
        )

        avg_latency = mean(
            [
                result["inference_seconds"]
                for result
                in scenario_results
            ]
        )

        summary_rows.append(
            {
                "scenario": scenario,
                "snr_db": (
                    scenario_results[0][
                        "snr_db"
                    ]
                ),
                "samples": len(
                    scenario_results
                ),
                "average_wer": avg_wer,
                "average_cer": avg_cer,
                "average_latency": (
                    avg_latency
                ),
                "wer_delta": (
                    avg_wer
                    - baseline_wer
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            999
            if row["scenario"] == "clean"
            else int(row["snr_db"])
        ),
        reverse=True,
    )

    print()
    print("=" * 78)

    print(
        f"{'Scenario':<15}"
        f"{'Samples':>9}"
        f"{'WER':>10}"
        f"{'CER':>10}"
        f"{'Latency':>12}"
        f"{'Δ WER':>10}"
    )

    print("-" * 78)

    for row in summary_rows:
        print(
            f"{row['scenario']:<15}"
            f"{row['samples']:>9}"
            f"{row['average_wer']:>10.2%}"
            f"{row['average_cer']:>10.2%}"
            f"{row['average_latency']:>10.2f}s"
            f"{row['wer_delta']:>10.2%}"
        )

    print("=" * 78)

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "snr_db",
                "samples",
                "average_wer",
                "average_cer",
                "average_latency",
                "wer_delta",
            ],
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print()
    print(
        f"Detailed results: {RESULTS_PATH}"
    )

    print(
        f"Summary:          {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()