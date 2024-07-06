def launch(
    rank: int,
    world_size: int,
    host: str,
    port: int,
    backend: str = "nccl",
    local_rank: int = None,
    seed: int = 1024,
    verbose: bool = True,
):
    """This function first parses the configuration arguments, using :func:`parse_args()` in case one of the input
    arguments are not given. Then initialize and set distributed environment by calling global_context's functions.

    Args:
        config (Union[str, dict, Config]): Config file or config file path are both acceptable
        rank (int): Rank for the default process group
        world_size (int): World size of the default process group
        host (str): The master address for distributed training
        port (str): The master port for distributed training
        backend (str, optional): Backend for ``torch.distributed``, defaults to ``nccl``
        local_rank (int, optional):
            Rank for the process on the node and is used to set the default CUDA device,
            defaults to None. If local_rank = None, the default device ordinal will be calculated automatically.
        seed (int, optional): Specified random seed for every process. Defaults to 1024.
        verbose (bool, optional): Whether to print logs. Defaults to True.

    Raises:
        Exception: Raise exception when config type is wrong
    """

    cur_accelerator = get_accelerator()

    backend = cur_accelerator.communication_backend

    # init default process group
    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend, init_method=init_method)

    # set cuda device
    # if local rank is not given, calculate automatically
    if cur_accelerator.support_set_device:
        cur_accelerator.set_device(local_rank)

    set_seed(seed)

    if verbose:
        logger = get_dist_logger()
        logger.info(f"Distributed environment is initialized, world size: {dist.get_world_size()}", ranks=[0])

