def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other XPU application.

    .. note::
        :func:`~torch.xpu.empty_cache` doesn't increase the amount of XPU
        memory available for PyTorch. However, it may help reduce fragmentation
        of XPU memory in certain cases.
    """
    if is_initialized():
        torch._C._xpu_emptyCache()

