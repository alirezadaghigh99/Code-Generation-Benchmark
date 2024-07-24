class CPUOffloadPolicy(OffloadPolicy):
    """
    This offload policy offloads parameters, gradients, and optimizer states to
    CPU. Sharded parameters are copied host-to-device before all-gather. The
    all-gathered parameters are freed according to ``reshard_after_forward``.
    Sharded gradients are copied device-to-host in backward, and the optimizer
    step runs on CPU with CPU optimizer states.

    Attributes:
        pin_memory (bool): Whether to pin sharded parameter and gradient
            memory. Pinning memory allows H2D/D2H copying without blocking the
            CPU and in turn, overlap with compute, but pinned memory cannot be
            used by other processes. Set this to ``False`` if you have
            insufficient CPU memory. (Default: ``True``)
    """

    pin_memory: bool = True

