import pytest

pytestmark = pytest.mark.evaluation

from src.evaluation.quality_gate import (
    evaluate_all_gates,
    load_summary,
)


SUMMARY_PATH = (
    "reports/noise_summary.csv"
)

CONFIG_PATH = (
    "config/quality_gates.json"
)


def test_all_quality_gates_pass():

    results = evaluate_all_gates(
        SUMMARY_PATH,
        CONFIG_PATH,
    )

    failures = []

    for result in results:

        if not result.passed:

            failures.append(
                (
                    f"{result.scenario}: "
                    + "; ".join(
                        result.reasons
                    )
                )
            )

    assert not failures, (
        "\nQuality gate failures:\n"
        + "\n".join(failures)
    )


def test_accuracy_degrades_with_noise():

    results = load_summary(
        SUMMARY_PATH
    )

    clean = results[
        "clean"
    ]["average_wer"]

    snr_20 = results[
        "snr_20db"
    ]["average_wer"]

    snr_10 = results[
        "snr_10db"
    ]["average_wer"]

    snr_5 = results[
        "snr_5db"
    ]["average_wer"]

    assert (
        clean
        <= snr_20
        <= snr_10
        <= snr_5
    ), (
        "Expected ASR quality to degrade "
        "as SNR decreases."
    )


def test_severe_noise_has_measurable_impact():

    results = load_summary(
        SUMMARY_PATH
    )

    clean_wer = results[
        "clean"
    ]["average_wer"]

    severe_noise_wer = results[
        "snr_5db"
    ]["average_wer"]

    assert severe_noise_wer > clean_wer