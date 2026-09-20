import json
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio.perturbations import (
    add_white_noise_at_snr,
)


SOURCE_MANIFEST = Path(
    "datasets/metadata/golden_manifest.json"
)

OUTPUT_ROOT = Path(
    "datasets/generated"
)

OUTPUT_MANIFEST = Path(
    "datasets/metadata/noise_manifest.json"
)

SNR_LEVELS = [
    20,
    10,
    5,
]

RANDOM_SEED = 42


def main():
    with SOURCE_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        samples = json.load(f)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    generated_samples = []

    print()
    print("Generating controlled noise scenarios...")
    print()

    for snr_db in SNR_LEVELS:

        scenario_name = f"snr_{snr_db}db"

        output_dir = (
            OUTPUT_ROOT / scenario_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Generating {scenario_name}..."
        )

        for index, sample in enumerate(
            samples,
            start=1,
        ):
            clean_audio, sample_rate = sf.read(
                sample["audio_file"],
                dtype="float32",
            )

            noisy_audio = add_white_noise_at_snr(
                clean_audio,
                snr_db=snr_db,
                rng=rng,
            )

            output_filename = (
                f"{sample['id']}_{scenario_name}.wav"
            )

            output_path = (
                output_dir / output_filename
            )

            sf.write(
                output_path,
                noisy_audio,
                sample_rate,
            )

            generated_samples.append(
                {
                    "id": (
                        f"{sample['id']}_"
                        f"{scenario_name}"
                    ),
                    "source_id": sample["id"],
                    "audio_file": (
                        str(output_path)
                        .replace("\\", "/")
                    ),
                    "expected_transcript": (
                        sample[
                            "expected_transcript"
                        ]
                    ),
                    "scenario": scenario_name,
                    "snr_db": snr_db,
                    "speaker_id": sample[
                        "speaker_id"
                    ],
                }
            )

            print(
                f"  [{index:02}/{len(samples)}] "
                f"{sample['id']}"
            )

        print()

    with OUTPUT_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            generated_samples,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 60)

    print(
        f"Generated {len(generated_samples)} "
        f"noise test cases."
    )

    print(
        f"Manifest: {OUTPUT_MANIFEST}"
    )


if __name__ == "__main__":
    main()