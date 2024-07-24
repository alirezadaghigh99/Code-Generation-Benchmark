class Device:
    """Describes device type and device id
    syntax: device_type:device_id(optional)
    example: 'CPU', 'CUDA', 'CUDA:1'
    """

    def __init__(self, device: str) -> None:
        options = device.split(":")
        self.type = getattr(DeviceType, options[0])
        self.device_id = 0
        if len(options) > 1:
            self.device_id = int(options[1])

