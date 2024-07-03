def set_rng_seed(rng_seed):
    """
    Sets seeds of `torch` and `torch.cuda` (if available).

    :param int rng_seed: The seed value.
    """
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)def set_rng_seed(rng_seed):
    """
    Sets seeds of `torch` and `torch.cuda` (if available).

    :param int rng_seed: The seed value.
    """
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)