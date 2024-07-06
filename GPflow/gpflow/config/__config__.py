def default_float() -> type:
    """Returns default float type"""
    return config().float

def default_jitter() -> float:
    """
    The jitter is a constant that GPflow adds to the diagonal of matrices
    to achieve numerical stability of the system when the condition number
    of the associated matrices is large, and therefore the matrices nearly singular.
    """
    return config().jitter

def as_context(temporary_config: Optional[Config] = None) -> Generator[None, None, None]:
    """Ensure that global configs defaults, with a context manager. Useful for testing."""
    current_config = config()
    temporary_config = replace(current_config) if temporary_config is None else temporary_config
    try:
        set_config(temporary_config)
        yield
    finally:
        set_config(current_config)

