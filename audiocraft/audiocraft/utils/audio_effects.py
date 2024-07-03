def select_audio_effects(
    audio_effects: tp.Dict,
    weights: tp.Optional[tp.Dict] = None,
    mode: str = "all",
    max_length: tp.Optional[int] = None,
):
    """Samples a subset of audio effects methods from the `AudioEffects` class.

    This function allows you to select a subset of audio effects
    based on the chosen selection mode and optional weights.

    Args:
        audio_effects (dict): A dictionary of available audio augmentations, usually
            obtained from the output of the 'get_audio_effects' function.
        weights (dict): A dictionary mapping augmentation names to their corresponding
            probabilities of being selected. This argument is used when 'mode' is set
            to "weighted." If 'weights' is None, all augmentations have equal
            probability of being selected.
        mode (str): The selection mode, which can be one of the following:
            - "all": Select all available augmentations.
            - "weighted": Select augmentations based on their probabilities in the
              'weights' dictionary.
        max_length (int): The maximum number of augmentations to select. If 'max_length'
            is None, no limit is applied.

    Returns:
        dict: A subset of the 'audio_effects' dictionary containing the selected audio
        augmentations.

    Note:
        - In "all" mode, all available augmentations are selected.
        - In "weighted" mode, augmentations are selected with a probability
          proportional to their weights specified in the 'weights' dictionary.
        - If 'max_length' is set, the function limits the number of selected
          augmentations.
        - If no augmentations are selected or 'audio_effects' is empty, the function
          defaults to including an "identity" augmentation.
        - The "identity" augmentation means that no audio effect is applied.
    """
    if mode == "all":  # original code
        out = audio_effects
    elif mode == "weighted":
        # Probability proportionnal to weights
        assert weights is not None
        out = {
            name: value
            for name, value in audio_effects.items()
            if random.random() < weights.get(name, 1.0)
        }
    else:
        raise ValueError(f"Unknown mode {mode}")
    if max_length is not None:
        # Help having a deterministic limit of the gpu memory usage
        random_keys = random.sample(list(out.keys()), max_length)
        out = {key: out[key] for key in random_keys}
    if len(out) == 0:  # Check not to return empty dict
        out = {"identity": AudioEffects.identity}
    return out