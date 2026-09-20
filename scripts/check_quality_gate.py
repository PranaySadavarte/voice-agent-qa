from src.evaluation.quality_gate import (
    evaluate_all_gates,
)


SUMMARY_PATH = (
    "reports/noise_summary.csv"
)

CONFIG_PATH = (
    "config/quality_gates.json"
)


def main():

    results = evaluate_all_gates(
        SUMMARY_PATH,
        CONFIG_PATH,
    )

    print()
    print("=" * 74)
    print("VOICE AI QUALITY GATE")
    print("=" * 74)

    overall_pass = True

    for result in results:

        status = (
            "PASS"
            if result.passed
            else "FAIL"
        )

        print()
        print(
            f"[{status}] "
            f"{result.scenario}"
        )

        print(
            "  WER:     "
            f"{result.current_wer:.2%} "
            f"(max {result.max_allowed_wer:.2%})"
        )

        print(
            "  Latency: "
            f"{result.current_latency:.2f}s "
            f"(max "
            f"{result.max_allowed_latency:.2f}s)"
        )

        if not result.passed:

            overall_pass = False

            for reason in result.reasons:

                print(
                    f"  ERROR: {reason}"
                )

    print()
    print("=" * 74)

    if overall_pass:

        print(
            "QUALITY GATE: PASS"
        )

    else:

        print(
            "QUALITY GATE: FAIL"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()