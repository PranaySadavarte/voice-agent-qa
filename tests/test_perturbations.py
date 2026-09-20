import numpy as np
import pytest

from src.audio.perturbations import (
    add_white_noise_at_snr,
    calculate_snr_db,
)


def test_noise_preserves_audio_shape():
    rng = np.random.default_rng(42)

    clean = np.ones(
        16000,
        dtype=np.float32,
    ) * 0.05

    noisy = add_white_noise_at_snr(
        clean,
        snr_db=20,
        rng=rng,
    )

    assert noisy.shape == clean.shape


def test_noise_is_deterministic_with_same_seed():
    clean = np.ones(
        16000,
        dtype=np.float32,
    ) * 0.05

    noisy_a = add_white_noise_at_snr(
        clean,
        snr_db=10,
        rng=np.random.default_rng(42),
    )

    noisy_b = add_white_noise_at_snr(
        clean,
        snr_db=10,
        rng=np.random.default_rng(42),
    )

    assert np.allclose(
        noisy_a,
        noisy_b,
    )


def test_generated_noise_matches_target_snr():
    sample_rate = 16000

    t = np.arange(sample_rate) / sample_rate

    clean = (
        0.05
        * np.sin(
            2 * np.pi * 440 * t
        )
    ).astype(np.float32)

    noisy = add_white_noise_at_snr(
        clean,
        snr_db=10,
        rng=np.random.default_rng(42),
    )

    measured_snr = calculate_snr_db(
        clean,
        noisy,
    )

    assert measured_snr == pytest.approx(
        10,
        abs=0.2,
    )