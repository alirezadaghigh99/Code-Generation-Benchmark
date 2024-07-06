def convert_audio(wav: torch.Tensor, from_rate: float,
                  to_rate: float, to_channels: int) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels."""
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    wav = convert_audio_channels(wav, to_channels)
    return wav

def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """Convert audio to the given number of channels.

    Args:
        wav (torch.Tensor): Audio wave of shape [B, C, T].
        channels (int): Expected number of channels as output.
    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
    """
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, and the stream has multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file has
        # a single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file has
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav

