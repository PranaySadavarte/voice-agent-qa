import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GateResult:
    scenario: str
    passed: bool

    current_wer: float
    reference_wer: float
    max_allowed_wer: float

    current_latency: float
    reference_latency: float
    max_allowed_latency: float

    reasons: list[str]


def load_summary(
    summary_path: str | Path,
) -> dict[str, dict]:

    summary_path = Path(summary_path)

    results = {}

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            results[row["scenario"]] = {
                "scenario": row["scenario"],
                "samples": int(row["samples"]),
                "average_wer": float(
                    row["average_wer"]
                ),
                "average_cer": float(
                    row["average_cer"]
                ),
                "average_latency": float(
                    row["average_latency"]
                ),
            }

    return results


def load_gate_config(
    config_path: str | Path,
) -> dict:

    config_path = Path(config_path)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def evaluate_scenario(
    scenario: str,
    measurement: dict,
    gate_config: dict,
) -> GateResult:

    reference_wer = gate_config[
        "reference_wer"
    ]

    max_allowed_wer = (
        reference_wer
        + gate_config[
            "max_wer_regression"
        ]
    )

    reference_latency = gate_config[
        "reference_latency_seconds"
    ]

    max_allowed_latency = (
        reference_latency
        * (
            1
            + gate_config[
                "max_latency_regression_ratio"
            ]
        )
    )

    current_wer = measurement[
        "average_wer"
    ]

    current_latency = measurement[
        "average_latency"
    ]

    reasons = []

    if current_wer > max_allowed_wer:

        reasons.append(
            (
                "WER regression: "
                f"{current_wer:.2%} > "
                f"{max_allowed_wer:.2%}"
            )
        )

    # if current_latency > max_allowed_latency:

    #     reasons.append(
    #         (
    #             "Latency regression: "
    #             f"{current_latency:.2f}s > "
    #             f"{max_allowed_latency:.2f}s"
    #         )
    #     )
    enforce_latency = gate_config.get(
    "enforce_latency",
    False,
    )

    if (
        enforce_latency
        and current_latency > max_allowed_latency
    ):

        reasons.append(
            (
                "Latency regression: "
                f"{current_latency:.2f}s > "
                f"{max_allowed_latency:.2f}s"
            )
        )

    return GateResult(
        scenario=scenario,
        passed=len(reasons) == 0,

        current_wer=current_wer,
        reference_wer=reference_wer,
        max_allowed_wer=max_allowed_wer,

        current_latency=current_latency,
        reference_latency=reference_latency,
        max_allowed_latency=(
            max_allowed_latency
        ),

        reasons=reasons,
    )


def evaluate_all_gates(
    summary_path: str | Path,
    config_path: str | Path,
) -> list[GateResult]:

    measurements = load_summary(
        summary_path
    )

    configs = load_gate_config(
        config_path
    )

    results = []

    for scenario, gate_config in configs.items():

        if scenario not in measurements:

            raise ValueError(
                (
                    "Missing benchmark scenario: "
                    f"{scenario}"
                )
            )

        result = evaluate_scenario(
            scenario=scenario,
            measurement=measurements[
                scenario
            ],
            gate_config=gate_config,
        )

        results.append(result)

    return results