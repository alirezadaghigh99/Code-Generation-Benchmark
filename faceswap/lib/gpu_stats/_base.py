def set_exclude_devices(devices: list[int]) -> None:
    """ Add any explicitly selected GPU devices to the global list of devices to be excluded
    from use by Faceswap.

    Parameters
    ----------
    devices: list[int]
        list of GPU device indices to exclude

    Example
    -------
    >>> set_exclude_devices([0, 1]) # Exclude the first two GPU devices
    """
    logger = logging.getLogger(__name__)
    logger.debug("Excluding GPU indicies: %s", devices)
    if not devices:
        return
    _EXCLUDE_DEVICES.extend(devices)

