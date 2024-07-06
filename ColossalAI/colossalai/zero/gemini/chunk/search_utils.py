def search_chunk_configuration(
    model: nn.Module,
    search_range_m: float,
    search_interval: int,  # hidden size is the best value for the interval
    min_chunk_size_m: float = 32,
    filter_exlarge_params: bool = True,
    strict_ddp_flag: bool = False,
    process_group: Optional[ProcessGroup] = None,
    memstas: Optional[MemStats] = None,
) -> Tuple[Dict, int, int]:
    """search_chunk_configuration

    Search the chunk configuration for a model.

    Args:
        model (nn.Module): torch module
        search_range_m (float): searching range divided by 2^20.
        search_interval (int): searching interval.
        min_chunk_size_m (float, optional): the minimum size of a distributed chunk, divided by 2^20..
        filter_exlarge_params (bool, optional): filter extreme large parameters. Defaults to True.
        strict_ddp_flag (bool, optional): whether to enable the strict ddp mode.
            all parameters keep replicated in this mode.

    Returns:
        Tuple[Dict, int]: chunk config (a dict of dp_degree -> chunk init args) and its memory chunk waste in byte.
    """

    if memstas is not None:
        param_order = memstas.param_order()
    else:
        # build the param visited order right now
        param_order = OrderedParamGenerator()
        for p in model.parameters():
            param_order.append(p)

    search_range = round(search_range_m * 1024**2)
    min_chunk_size = round(min_chunk_size_m * 1024**2)
    assert search_range >= 0

    params_dict = classify_params_by_dp_degree(param_order, process_group)
    size_lcm = np.lcm.reduce(list(params_dict.keys()))
    config_dict: Dict[int, Dict] = dict()
    total_param_size = 0

    size_dict: Dict[int, List[int]] = dict()
    for dp_degree in params_dict:
        params_list = params_dict[dp_degree]
        size_list = [_tensor_numel(p) for p in params_list]
        group_acc_size = sum(size_list)
        total_param_size += group_acc_size

        # let small parameters keep gathered in CUDA all the time
        if group_acc_size < min_chunk_size:
            config_dict[dp_degree] = dict(chunk_size=group_acc_size, keep_gathered=True)
        else:
            size_dict[dp_degree] = size_list

    if filter_exlarge_params:
        _filter_exlarge_params(model, size_dict)

    max_size = min_chunk_size
    for key in size_dict:
        max_size = max(max_size, max(size_dict[key]))
    start_size = int(math.ceil(max_size / search_interval) * search_interval)

    min_chunk_waste = float("+inf")
    best_chunk_size = start_size

    for chunk_size in range(start_size, start_size + search_range + 1, search_interval):
        temp_waste = 0
        for key in size_dict:
            temp_waste += _get_unused_byte(size_dict[key], chunk_size)
        if temp_waste < min_chunk_waste:
            min_chunk_waste = temp_waste
            best_chunk_size = chunk_size

    # the chunk size needs to be divided by each groups sizes
    best_chunk_size = best_chunk_size + (-best_chunk_size % size_lcm)
    for dp_degree in params_dict:
        if dp_degree in config_dict:
            continue
        config_dict[dp_degree] = dict(chunk_size=best_chunk_size, keep_gathered=False)

    return config_dict, total_param_size, min_chunk_waste

