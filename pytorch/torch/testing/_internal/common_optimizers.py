def _get_optim_inputs_including_global_cliquey_kwargs(
    device, dtype, optim_info, skip=()
) -> List[OptimizerInput]:
    """
    Return a list of all configs for a given optimizer as a list of OptimizerInputs,
    including configs that have supported global cliquey kwargs (foreach, fused,
    differentiable) based on optim_info.supported_impls.

    The configs (optim_inputs) returned by optim_info.optim_inputs_func(...)
    intentionally do NOT include global cliquey kwargs to give flexibility to tests.
    For example, testing correctness between toggling foreach on and off is now
    trivial. That said, we sometimes want to test for all possible configs on an
    optimizer including all supported flags, so this helper returns all optim inputs.
    """
    assert all(
        x in ["foreach", "fused", "differentiable"] for x in skip
    ), "skip must be a subset of ['foreach', 'fused', 'differentiable']"

    optim_inputs = optim_info.optim_inputs_func(device)

    supported_impls = tuple(
        x
        for x in optim_info.supported_impls
        if x not in skip
        and (_get_device_type(device) in optim_info.supports_fused_on or x != "fused")
        and (
            _get_device_type(device) in _get_foreach_kernels_supported_devices()
            or x != "foreach"
        )
    )

    all_optim_inputs = []
    for optim_input in optim_inputs:
        # Add the base config where all the flags are False
        base_kwargs = deepcopy(optim_input.kwargs)
        if len(supported_impls) != 0:
            for flag in supported_impls:
                base_kwargs[flag] = False
            all_optim_inputs.append(
                OptimizerInput(params=None, kwargs=base_kwargs, desc=optim_input.desc)
            )
        else:
            all_optim_inputs.append(optim_input)
        # Add a config for when each of the global cliquey kwargs is True
        # Note that in [optimizer kwarg categories], these kwargs are mutually
        # exclusive, so we do not need to product them together.
        for flag in supported_impls:
            new_kwargs = deepcopy(base_kwargs)
            new_kwargs[flag] = True
            all_optim_inputs.append(
                OptimizerInput(
                    params=None, kwargs=new_kwargs, desc=f"{optim_input.desc} & {flag}"
                )
            )
    return all_optim_inputs

