def hash_for_seed(obj):
    """
    Hash any object into an int that can be used in `np.random.seed`, and does not change between Python sessions.

    Args:
      obj: any object with `repr` defined to show its states.
    """

    bs = repr(obj).encode()
    out = 0
    for b in bs:
        out = (out * 256 + b) % 4294967291  # Output in [0, 2**32 - 1]
    return out

class set_config:
    """
    Temporarily changes the value of the configuration `name`.

    Example:

    >>> with set_config("netket_experimental_disable_ode_jit", True):
    >>>     run_code

    """

    def __init__(self, name: str, value: Any):
        self._name = name.upper()
        self._value = value

    def __enter__(self):
        self._orig_value = nk.config.FLAGS[self._name]
        nk.config.update(self._name, self._value)

    def __exit__(self, exc_type, exc_value, traceback):
        nk.config.update(self._name, self._orig_value)

