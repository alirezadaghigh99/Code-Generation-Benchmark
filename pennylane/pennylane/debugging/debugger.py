class PLDB(pdb.Pdb):
    """Custom debugging class integrated with Pdb.

    This class is responsible for storing and updating a global device to be
    used for executing quantum circuits while in debugging context. The core
    debugger functionality is inherited from the native Python debugger (Pdb).

    This class is not directly user-facing, but is interfaced with the
    ``qml.breakpoint()`` function and ``pldb_device_manager`` context manager.
    The former is responsible for launching the debugger prompt and the latter
    is responsible with extracting and storing the ``qnode.device``.

    The device information is used for validation checks and to execute measurements.
    """

    __active_dev = None

    def __init__(self, *args, **kwargs):
        """Initialize the debugger, and set custom prompt string."""
        super().__init__(*args, **kwargs)
        self.prompt = "[pldb]: "

    @classmethod
    def valid_context(cls):
        """Determine if the debugger is called in a valid context.

        Raises:
            RuntimeError: breakpoint is called outside of a qnode execution
            TypeError: breakpoints not supported on this device
        """

        if not qml.queuing.QueuingManager.recording() or not cls.has_active_dev():
            raise RuntimeError("Can't call breakpoint outside of a qnode execution")

        if cls.get_active_device().name not in ("default.qubit", "lightning.qubit"):
            raise TypeError("Breakpoints not supported on this device")

    @classmethod
    def add_device(cls, dev):
        """Update the global active device.

        Args:
            dev (Union[Device, "qml.devices.Device"]): the active device
        """
        cls.__active_dev = dev

    @classmethod
    def get_active_device(cls):
        """Return the active device.

        Raises:
            RuntimeError: No active device to get

        Returns:
            Union[Device, "qml.devices.Device"]: The active device
        """
        if not cls.has_active_dev():
            raise RuntimeError("No active device to get")

        return cls.__active_dev

    @classmethod
    def has_active_dev(cls):
        """Determine if there is currently an active device.

        Returns:
            bool: True if there is an active device
        """
        return bool(cls.__active_dev)

    @classmethod
    def reset_active_dev(cls):
        """Reset the global active device variable to None."""
        cls.__active_dev = None

    @classmethod
    def _execute(cls, batch_tapes):
        """Execute the batch of tapes on the active device"""
        dev = cls.get_active_device()

        valid_batch = batch_tapes
        if dev.wires:
            valid_batch = qml.devices.preprocess.validate_device_wires(
                batch_tapes, wires=dev.wires
            )[0]

        program, new_config = dev.preprocess()
        new_batch, fn = program(valid_batch)

        # TODO: remove [0] index once compatible with transforms
        return fn(dev.execute(new_batch, new_config))[0]

