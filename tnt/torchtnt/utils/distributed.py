def spawn_multi_process(
    world_size: int,
    backend: str,
    method: Callable[TParams, TReturn],
    *method_args: Any,
    **method_kwargs: Any,
) -> List[TReturn]:
    """
    Spawn single node, multi-rank function.
    Uses localhost and free port to communicate.

    Args:
        world_size: number of processes
        backend: backend to use. for example, "nccl", "gloo", etc
        method: callable to spawn.
        method_args: args for the method
        method_kwargs: kwargs for the method

    Note:
        The default timeout used for distributed collectives in the process group is 60 seconds.
        This can be overridden by passing a `timeout_s` key in the `method_kwargs`. It will be
        extracted before passing to the method call.

    Returns:
        A list, l, where l[i] is the return value of method(*method_args, **methods_kwargs) on rank i
    """
    manager = multiprocessing.Manager()
    mp_output_dict = manager.dict()

    port = str(get_free_port())
    torch.multiprocessing.spawn(
        # torch.multiprocessing.spawn sends rank as the first param
        # https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn
        _init_pg_and_rank_and_launch_method,
        args=(
            ProcessGroupSetupParams(
                backend=backend,
                port=port,
                world_size=world_size,
                timeout_s=method_kwargs.pop("timeout_s", 60),
            ),
            mp_output_dict,
            method,
            method_args,
            method_kwargs,
        ),
        nprocs=world_size,
    )

    output_list = []
    for i in range(world_size):
        output_list.append(mp_output_dict[i])

    return output_list

def get_global_rank() -> int:
    """
    Get rank using torch.distributed if available. Otherwise, the RANK env var instead if initialized.
    Returns 0 if neither condition is met.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()

    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(os.environ["RANK"])

    return 0

def all_gather_tensors(
    result: Tensor, group: Optional[dist.ProcessGroup] = None
) -> List[Tensor]:
    """Function to gather tensors from several distributed processes onto a list that is broadcasted to all processes.
    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    # if torch.distributed is not available or not initialized
    # return single-item list containing the result
    if not dist.is_available() or not dist.is_initialized():
        return [result]

    # convert tensors to contiguous format
    result = result.contiguous()
    world_size = dist.get_world_size(group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_all_gather_tensors(result, group, world_size)

    # gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    stacked_local_size = [world_size] + list(local_size.size())
    local_sizes = list(
        torch.zeros(
            stacked_local_size, dtype=local_size.dtype, device=local_size.device
        )
    )
    dist.all_gather(local_sizes, local_size, group=group)

    # if the backend is NCCL, we can gather the differently sized tensors without padding
    if dist.get_backend(group) == "nccl":
        gathered_result = [result.new_empty(size.tolist()) for size in local_sizes]
        dist.all_gather(gathered_result, result, group)
        return gathered_result

    # if shapes are all the same, then do a simple gather:
    stacked_sizes = torch.stack(local_sizes)
    max_size = stacked_sizes.max(dim=0).values
    min_size = stacked_sizes.min(dim=0).values
    all_sizes_equal = torch.equal(max_size, min_size)
    if all_sizes_equal:
        return _simple_all_gather_tensors(result, group, world_size)

    # if not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    stacked_result_padded = [world_size] + list(result_padded.size())
    gathered_result = list(
        torch.zeros(
            stacked_result_padded,
            dtype=result_padded.dtype,
            device=result_padded.device,
        )
    )
    dist.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result

def get_process_group_backend_from_device(device: torch.device) -> str:
    """Function that gets the default process group backend from the device."""
    return "nccl" if device.type == "cuda" else "gloo"

def get_world_size() -> int:
    """
    Get world size using torch.distributed if available. Otherwise, the WORLD_SIZE env var is used instead if initialized.
    Returns 1 if neither condition is met.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()

    world_size = os.environ.get("WORLD_SIZE", "")
    if world_size.isdecimal():
        return int(world_size)

    return 1

def sync_bool(
    val: bool,
    pg: Optional[dist.ProcessGroup] = None,
    coherence_mode: Union[Literal["any", "all", "rank_zero"], int, float] = "any",
) -> bool:
    """Utility to synchronize a boolean value across members of a provided process group.

    In the case ``torch.distributed`` is not available or initialized, the input ``val`` is returned.

    Args:
        val (bool): boolean value to synchronize
        pg: process group to use for synchronization. If not specified, the default process group is used.
        coherence_mode Union[str, int, float]: the manner in which the boolean value should be synchronized. 5 options are currently supported:
            1. any (default): If any rank provides a True value, all ranks should receive True.
            2. all: Only if all ranks provide a True value should all ranks receive True.
            3. rank_zero: Makes rank 0 process's value the source of truth and broadcasts the result to all other processes.
            4. If an integer N is provided, return True only if at least N processes provide a True value.
            5. If a float F is provided, return True only if at least this ratio of processes provide a True value. The ratio provided should be in the range [0, 1].

    Returns:
        The synchronized boolean value.

    Example::

        >>> val = True
        >>> # synced_val is True iff all ranks provide a True value to the function
        >>> synced_val = sync_bool(val, coherence_mode="all")
        >>> if synced_val:
        >>>     print("success")

    """
    if not dist.is_available() or not dist.is_initialized():
        return val

    pg = pg or dist.group.WORLD
    device = torch.device(
        torch.cuda.current_device() if dist.get_backend(pg) == "nccl" else "cpu"
    )
    pg_wrapper = PGWrapper(pg)

    dtype = torch.uint8
    if pg_wrapper.get_world_size() > 256:
        dtype = torch.int

    indicator = (
        torch.ones(1, device=device, dtype=dtype)
        if val
        else torch.zeros(1, device=device, dtype=dtype)
    )

    if coherence_mode == "rank_zero":
        # Broadcast from rank 0 to all other ranks
        dist.broadcast(indicator, src=0, group=pg)
        return bool(indicator[0].item())
    elif coherence_mode == "any":
        # sum up the indicators across all the ranks.
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return indicator.item() > 0
    elif coherence_mode == "all":
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return indicator.item() == pg_wrapper.get_world_size()
    elif isinstance(coherence_mode, int):
        # if >= int(coherence_mode) processes signal to stop, all processes stop
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return indicator.item() >= coherence_mode
    elif isinstance(coherence_mode, float):
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return (indicator.item() / pg_wrapper.get_world_size()) >= coherence_mode
    else:
        raise TypeError(
            f'Invalid value for `coherence_mode` provided: Expected type int, float, or one of ("any", "all", "rank_zero"), but received {coherence_mode}.'
        )

