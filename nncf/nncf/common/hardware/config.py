def get_hw_config_type(target_device: str) -> Optional[HWConfigType]:
    """
    Returns hardware configuration type for target device

    :param target_device: A target device
    :raises ValueError: if target device is not supported yet
    :return: hardware configuration type or None for the 'TRIAL' target device
    """
    if target_device == "TRIAL":
        return None
    return HWConfigType(HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device])

