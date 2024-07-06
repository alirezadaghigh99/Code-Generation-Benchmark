def set_seeds(seed: int):
    """Set seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed to be set
    """
    if seed is not None and isinstance(seed, int):
        np.random.seed(seed)
        random.seed(seed)
        if 'torch' in sys.modules:
            import torch  # pylint: disable=import-outside-toplevel
            torch.manual_seed(seed)
        if 'tensorflow' in sys.modules:
            import tensorflow as tf  # pylint: disable=import-outside-toplevel
            tf.random.set_seed(seed)

