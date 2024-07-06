def shuffle(
    a: np.ndarray,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """Shuffles an array in-place, using a specified random state or creating a new one if not provided.

    Args:
        a (np.ndarray): The array to be shuffled.
        random_state (Optional[np.random.RandomState], optional): The random state used for shuffling. Defaults to None.

    Returns:
        np.ndarray: The shuffled array (note: the shuffle is in-place, so the original array is modified).
    """
    if random_state is None:
        random_state = get_random_state()
    random_state.shuffle(a)
    return a

