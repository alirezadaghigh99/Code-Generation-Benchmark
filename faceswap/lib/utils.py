def get_backend() -> ValidBackends:
    """ Get the backend that Faceswap is currently configured to use.

    Returns
    -------
    str
        The backend configuration in use by Faceswap. One of  ["cpu", "directml", "nvidia", "rocm",
        "apple_silicon"]

    Example
    -------
    >>> from lib.utils import get_backend
    >>> get_backend()
    'nvidia'
    """
    return _FS_BACKEND

