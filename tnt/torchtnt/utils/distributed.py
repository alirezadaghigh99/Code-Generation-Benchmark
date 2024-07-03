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
            ProcessGroupSetupParams(backend=backend, port=port, world_size=world_size),
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

    return output_listdef get_global_rank() -> int:
    """
    Get rank using torch.distributed if available. Otherwise, the RANK env var instead if initialized.
    Returns 0 if neither condition is met.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()

    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(os.environ["RANK"])

    return 0def all_gather_tensors(
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
    return gathered_resultdef get_process_group_backend_from_device(device: torch.device) -> str:
    """Function that gets the default process group backend from the device."""
    return "nccl" if device.type == "cuda" else "gloo"def get_world_size() -> int:
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