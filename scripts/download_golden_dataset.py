import json
import shutil
from pathlib import Path

from datasets import Audio, load_dataset


DATASET_NAME = "sanchit-gandhi/librispeech_asr_dummy"
CONFIG = "default"
SPLIT = "test.clean"
NUM_SAMPLES = 15

OUTPUT_DIR = Path("datasets/clean")
MANIFEST_PATH = Path("datasets/metadata/golden_manifest.json")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading LibriSpeech golden dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        CONFIG,
        split=SPLIT,
    )

    # We only need the encoded FLAC file, not Hugging Face audio decoding.
    dataset = dataset.cast_column(
        "audio",
        Audio(decode=False),
    )

    sample_count = min(NUM_SAMPLES, len(dataset))
    dataset = dataset.select(range(sample_count))

    manifest = []

    for index, sample in enumerate(dataset):
        sample_id = sample["id"]

        audio_info = sample["audio"]

        filename = f"{sample_id}.flac"
        output_path = OUTPUT_DIR / filename

        if audio_info["bytes"] is not None:
            output_path.write_bytes(audio_info["bytes"])

        elif audio_info["path"] is not None:
            shutil.copy2(
                audio_info["path"],
                output_path,
            )

        else:
            raise RuntimeError(
                f"No audio data found for sample {sample_id}"
            )

        entry = {
            "id": sample_id,
            "audio_file": str(output_path).replace("\\", "/"),
            "expected_transcript": sample["text"],
            "scenario": "clean",
            "speaker_id": sample["speaker_id"],
        }

        manifest.append(entry)

        print(
            f"[{index + 1:02}/{sample_count}] "
            f"{sample_id}: {sample['text'][:60]}"
        )

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Saved {len(manifest)} golden samples.")
    print(f"Audio:    {OUTPUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()