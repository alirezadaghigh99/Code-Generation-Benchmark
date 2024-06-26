class GPUInfo():
    """Dataclass for storing information about the available GPUs on the system.

    Attributes:
    ----------
    vram: list[int]
        List of integers representing the total VRAM available on each GPU, in MB.
    vram_free: list[int]
        List of integers representing the free VRAM available on each GPU, in MB.
    driver: str
        String representing the driver version being used for the GPUs.
    devices: list[str]
        List of strings representing the names of each GPU device.
    devices_active: list[int]
        List of integers representing the indices of the active GPU devices.
    """
    vram: list[int]
    vram_free: list[int]
    driver: str
    devices: list[str]
    devices_active: list[int]