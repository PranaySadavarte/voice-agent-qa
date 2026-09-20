def calculate_rtf(
    inference_seconds: float,
    audio_duration_seconds: float,
) -> float:

    if audio_duration_seconds <= 0:
        raise ValueError(
            "Audio duration must be positive."
        )

    return (
        inference_seconds
        / audio_duration_seconds
    )