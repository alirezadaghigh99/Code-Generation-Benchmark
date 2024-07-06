def get_module_summary(
    module: torch.nn.Module,
    # pyre-fixme
    module_args: Optional[Tuple[Any, ...]] = None,
    # pyre-fixme
    module_kwargs: Optional[MutableMapping[str, Any]] = None,
) -> ModuleSummary:
    """
    Generate a :class:`~ModuleSummary` object, then assign its values and generate submodule tree.

    Args:
        module: The module to be summarized.
        module_args: A tuple of arguments for the module to run and calculate FLOPs and activation sizes.
        module_kwargs: Any kwarg arguments to be passed into the module's forward function.

            Note:
              To calculate FLOPs, you must use PyTorch 1.13 or greater.

            Note:
              If module contains any lazy submodule, we will NOT calculate FLOPs.

            Note:
              Currently only modules that output a single tensor are supported.
              TODO: to support more flexible output for module.

    """

    module_summary_data = _ModuleSummaryData()
    has_uninitialized_param = _has_uninitialized_param(module)
    if not has_uninitialized_param:
        has_tensor_in_args = _has_tensor(module_args)
        has_tensor_in_kwargs = _has_tensor(module_kwargs)
        if has_tensor_in_kwargs:
            warnings.warn(
                "A tensor in module_kwargs was found. This may lead to an inaccurately computed activation size, as keyword arguments are not passed into forward hooks for modules. "
                "For best results, please input tensors though module_args."
            )
        if has_tensor_in_args or has_tensor_in_kwargs:
            module_summary_data = _get_module_flops_and_activation_sizes(
                module, module_args, module_kwargs
            )

    return _generate_module_summary(module, "", module_summary_data)

