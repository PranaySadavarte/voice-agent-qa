import numpy as np


def add_white_noise_at_snr(
    audio: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add Gaussian white noise to audio at a specified
    signal-to-noise ratio (SNR).

    Higher SNR = cleaner audio.
    Lower SNR = noisier audio.
    """

    audio = np.asarray(audio, dtype=np.float32)

    signal_power = np.mean(audio ** 2)

    if signal_power <= 0:
        raise ValueError("Audio contains no measurable signal.")

    target_noise_power = (
        signal_power / (10 ** (snr_db / 10))
    )

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=audio.shape,
    ).astype(np.float32)

    current_noise_power = np.mean(noise ** 2)

    noise *= np.sqrt(
        target_noise_power / current_noise_power
    )

    noisy_audio = audio + noise

    # Prevent clipping while preserving SNR.
    peak = np.max(np.abs(noisy_audio))

    if peak > 0.99:
        noisy_audio *= 0.99 / peak

    return noisy_audio.astype(np.float32)


def calculate_snr_db(
    clean_audio: np.ndarray,
    noisy_audio: np.ndarray,
) -> float:
    """
    Calculate measured SNR between a clean signal
    and its noisy version.

    Intended primarily for validation/tests where no
    peak rescaling occurred.
    """

    clean_audio = np.asarray(
        clean_audio,
        dtype=np.float64,
    )

    noisy_audio = np.asarray(
        noisy_audio,
        dtype=np.float64,
    )

    if clean_audio.shape != noisy_audio.shape:
        raise ValueError(
            "Clean and noisy audio must have the same shape."
        )

    noise = noisy_audio - clean_audio

    signal_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise ** 2)

    if signal_power <= 0:
        raise ValueError("Clean audio has zero power.")

    if noise_power <= 0:
        return float("inf")

    return float(
        10 * np.log10(
            signal_power / noise_power
        )
    )