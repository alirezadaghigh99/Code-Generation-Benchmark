def seed(seed: int, deterministic: Optional[Union[str, int]] = None) -> None:
    """Function that sets seed for pseudo-random number generators across commonly used libraries.

    This seeds PyTorch, NumPy, and the python.random module.
    For more details, see https://pytorch.org/docs/stable/notes/randomness.html.

    Args:
        seed: the integer value seed.
        deterministic: Controls determinism settings within PyTorch.
            If `None`, don't set any PyTorch global values.
            If "default" or 0, don't error or warn on nondeterministic operations and additionally enable PyTorch CuDNN benchmark.
            If "warn" or 1, warn on nondeterministic operations and disable PyTorch CuDNN benchmark.
            If "error" or 2, error on nondeterministic operations and disable PyTorch CuDNN benchmark.
            For more details, see :func:`torch.set_deterministic_debug_mode` and
            https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms.

    Raises:
        ValueError
            If the input seed value is outside the required range.
    """
    max_val = np.iinfo(np.uint32).max
    min_val = np.iinfo(np.uint32).min
    if seed < min_val or seed > max_val:
        raise ValueError(
            f"Invalid seed value provided: {seed}. Value must be in the range [{min_val}, {max_val}]"
        )
    _log.debug(f"Setting seed to {seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic is not None:
        _log.debug(f"Setting deterministic debug mode to {deterministic}")
        torch.set_deterministic_debug_mode(deterministic)
        deterministic_debug_mode = torch.get_deterministic_debug_mode()
        if deterministic_debug_mode == 0:
            _log.debug("Disabling cuDNN deterministic mode")
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            _log.debug("Enabling cuDNN deterministic mode")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # reference: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

