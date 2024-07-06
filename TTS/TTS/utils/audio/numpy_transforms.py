def build_mel_basis(
    *,
    sample_rate: int = None,
    fft_size: int = None,
    num_mels: int = None,
    mel_fmax: int = None,
    mel_fmin: int = None,
    **kwargs,
) -> np.ndarray:
    """Build melspectrogram basis.

    Returns:
        np.ndarray: melspectrogram basis.
    """
    if mel_fmax is not None:
        assert mel_fmax <= sample_rate // 2
        assert mel_fmax - mel_fmin > 0
    return librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=mel_fmin, fmax=mel_fmax)

