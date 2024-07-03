def copy_data_to_device(data: T, device: torch.device, *args: Any, **kwargs: Any) -> T:
    """Function that recursively copies data to a torch.device.

    Args:
        data: The data to copy to device
        device: The device to which the data should be copied
        args: positional arguments that will be passed to the `to` call
        kwargs: keyword arguments that will be passed to the `to` call

    Returns:
        The data on the correct device
    """

    # Redundant isinstance(data, tuple) check is required here to make pyre happy
    if _is_named_tuple(data) and isinstance(data, tuple):
        return type(data)(
            **copy_data_to_device(data._asdict(), device, *args, **kwargs)
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(copy_data_to_device(e, device, *args, **kwargs) for e in data)
    elif isinstance(data, defaultdict):
        return type(data)(
            data.default_factory,
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            },
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            }
        )
    elif is_dataclass(data) and not isinstance(data, type):
        new_data_class = type(data)(
            **{
                field.name: copy_data_to_device(
                    getattr(data, field.name), device, *args, **kwargs
                )
                for field in fields(data)
                if field.init
            }
        )
        for field in fields(data):
            if not field.init:
                setattr(
                    new_data_class,
                    field.name,
                    copy_data_to_device(
                        getattr(data, field.name), device, *args, **kwargs
                    ),
                )
        return new_data_class
    elif isinstance(data, _CopyableData):
        return data.to(device, *args, **kwargs)
    return datadef record_data_in_stream(data: T, stream: torch.cuda.streams.Stream) -> None:
    """
    Records the tensor element on certain streams, to avoid memory from being reused for another tensor.
    As mentioned in
    https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html, PyTorch
    uses the "caching allocator" for memory allocation for tensors. When a tensor is
    freed, its memory is likely to be reused by newly constructed tensors. By default,
    this allocator traces whether a tensor is still in use by only the CUDA stream where
    it was created. When a tensor is used by additional CUDA streams, we need to call
    `record_stream` to tell the allocator about these streams. Otherwise, the allocator
    might free the underlying memory of the tensor once it is no longer used by the
    creator stream. This is a notable programming trick when we write programs using
    multiple CUDA streams.

    Args:
        data: The data on which to call record_stream
        stream: The CUDA stream with which to call record_stream
    """

    # Redundant isinstance(data, tuple) check is required here to make pyre happy
    if _is_named_tuple(data) and isinstance(data, tuple):
        record_data_in_stream(data._asdict(), stream)
    elif isinstance(data, (list, tuple)):
        for e in data:
            record_data_in_stream(e, stream)
    elif isinstance(data, Mapping):
        for _, v in data.items():
            record_data_in_stream(v, stream)
    elif is_dataclass(data) and not isinstance(data, type):
        for field in fields(data):
            record_data_in_stream(getattr(data, field.name), stream)
    elif isinstance(data, _MultistreamableData):
        data.record_stream(stream)def get_device_from_env() -> torch.device:
    """Function that gets the torch.device based on the current environment.

    This currently supports only CPU, GPU, and MPS devices. If CUDA is available, this function also sets the CUDA device.

    Within a distributed context, this function relies on the ``LOCAL_RANK`` environment variable
    to be made available by the program launcher for setting the appropriate device index.

    Raises:
        RuntimeError
            If ``LOCAL_RANK`` is outside the range of available GPU devices.
    """
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                "The local rank is larger than the number of available GPUs."
            )
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device