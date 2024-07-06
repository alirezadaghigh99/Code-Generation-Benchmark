def get_accelerator() -> BaseAccelerator:
    """
    Return the accelerator for the current process. If the accelerator is not initialized, it will be initialized
    to the default accelerator type.

    Returns: the accelerator for the current process.
    """
    global _ACCELERATOR

    if _ACCELERATOR is None:
        auto_set_accelerator()
    return _ACCELERATOR

