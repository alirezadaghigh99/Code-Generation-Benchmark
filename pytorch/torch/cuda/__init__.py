def is_available() -> bool:
    r"""Return a bool indicating if CUDA is currently available."""
    if not _is_compiled():
        return False
    if _nvml_based_avail():
        # The user has set an env variable to request this availability check that attempts to avoid fork poisoning by
        # using NVML at the cost of a weaker CUDA availability assessment. Note that if NVML discovery/initialization
        # fails, this assessment falls back to the default CUDA Runtime API assessment (`cudaGetDeviceCount`)
        return device_count() > 0
    else:
        # The default availability inspection never throws and returns 0 if the driver is missing or can't
        # be initialized. This uses the CUDA Runtime API `cudaGetDeviceCount` which in turn initializes the CUDA Driver
        # API via `cuInit`
        return torch._C._cuda_getDeviceCount() > 0def device_count() -> int:
    r"""Return the number of GPUs available."""
    global _cached_device_count
    if not _is_compiled():
        return 0
    if _cached_device_count is not None:
        return _cached_device_count
    # bypass _device_count_nvml() if rocm (not supported)
    nvml_count = _device_count_amdsmi() if torch.version.hip else _device_count_nvml()
    r = torch._C._cuda_getDeviceCount() if nvml_count < 0 else nvml_count
    # NB: Do not cache the device count prior to CUDA initialization, because
    # the number of devices can change due to changes to CUDA_VISIBLE_DEVICES
    # setting prior to CUDA initialization.
    if _initialized:
        _cached_device_count = r
    return rclass device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch.cuda._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch.cuda._maybe_exchange_device(self.prev_idx)
        return Falsedef get_device_capability(device: Optional[_device_t] = None) -> Tuple[int, int]:
    r"""Get the cuda capability of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor cuda capability of the device
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor