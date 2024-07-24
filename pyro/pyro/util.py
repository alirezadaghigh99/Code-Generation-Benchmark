def set_rng_seed(rng_seed):
    """
    Sets seeds of `torch` and `torch.cuda` (if available).

    :param int rng_seed: The seed value.
    """
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)

def set_rng_seed(rng_seed):
    """
    Sets seeds of `torch` and `torch.cuda` (if available).

    :param int rng_seed: The seed value.
    """
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)

class optional:
    """
    Optionally wrap inside `context_manager` if condition is `True`.
    """

    def __init__(self, context_manager, condition):
        self.context_manager = context_manager
        self.condition = condition

    def __enter__(self):
        if self.condition:
            return self.context_manager.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.condition:
            return self.context_manager.__exit__(exc_type, exc_val, exc_tb)

