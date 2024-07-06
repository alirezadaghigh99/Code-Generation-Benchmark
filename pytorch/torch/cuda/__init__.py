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
        return torch._C._cuda_getDeviceCount() > 0

def device_count() -> int:
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
    return r

def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    streamdata = torch._C._cuda_getCurrentStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )

def get_device_capability(device: Optional[_device_t] = None) -> Tuple[int, int]:
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

def get_device_properties(device: _device_t) -> _CudaDeviceProperties:
    r"""Get the properties of a device.

    Args:
        device (torch.device or int or str): device for which to return the
            properties of the device.

    Returns:
        _CudaDeviceProperties: the properties of the device
    """
    _lazy_init()  # will define _get_device_properties
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _get_device_properties(device)  # type: ignore[name-defined]

