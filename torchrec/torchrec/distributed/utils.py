def merge_fused_params(
    fused_params: Optional[Dict[str, Any]] = None,
    param_fused_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Configure the fused_params including cache_precision if the value is not preset.

    Values set in table_level_fused_params take precidence over the global fused_params

    Args:
        fused_params (Optional[Dict[str, Any]]): the original fused_params
        grouped_fused_params

    Returns:
        [Dict[str, Any]]: a non-null configured fused_params dictionary to be
        used to configure the embedding lookup kernel
    """

    if fused_params is None:
        fused_params = {}
    if param_fused_params is None:
        param_fused_params = {}
    if "lr" in param_fused_params:
        param_fused_params["learning_rate"] = param_fused_params.pop("lr")

    _fused_params = copy.deepcopy(fused_params)
    _fused_params.update(param_fused_params)
    return _fused_params

