class DeviceGuard:
    """
    This class provides a context manager for device switching. This is a stripped
    down version of torch.{device_name}.device.

    The context manager changes the current device to the given device index
    on entering the context and restores the original device on exiting.
    The device is switched using the provided device interface.
    """

    def __init__(self, device_interface: Type[DeviceInterface], index: Optional[int]):
        self.device_interface = device_interface
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        if self.idx is not None:
            self.prev_idx = self.device_interface.exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        if self.idx is not None:
            self.idx = self.device_interface.maybe_exchange_device(self.prev_idx)
        return False

