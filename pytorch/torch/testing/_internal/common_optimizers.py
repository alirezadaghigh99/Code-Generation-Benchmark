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

class optims(_TestParametrizer):
    """Decorator for specifying a list of optimizers over which to run a test."""

    def __init__(self, optim_info_iterable, dtypes=None):
        self.optim_info_list = list(optim_info_iterable)

        # optimizers aren't limited to be one dtype as parameters can have different dtypes
        # We default to torch.float32, but dtypes should be specified through passed in
        # parameters.
        self.dtypes = dtypes if dtypes is not None else [torch.float32]

    def _parametrize_test(self, test, generic_cls, device_cls):
        if device_cls is None:
            raise RuntimeError(
                "The @optims decorator is only intended to be used in a device-specific "
                "context; use it with instantiate_device_type_tests() instead of "
                "instantiate_parametrized_tests()"
            )

        for optim_info, dtype in itertools.product(self.optim_info_list, self.dtypes):
            # Construct the test name; device / dtype parts are handled outside.
            # See [Note: device and dtype suffix placement]
            test_name = optim_info.name

            # Construct parameter kwargs to pass to the test.
            param_kwargs = {"optim_info": optim_info, "dtype": dtype}

            try:

                @functools.wraps(test)
                def test_wrapper(*args, **kwargs):
                    return test(*args, **kwargs)

                decorator_fn = functools.partial(
                    optim_info.get_decorators,
                    generic_cls.__name__,
                    test.__name__,
                    device_cls.device_type,
                    dtype,
                )

                yield (test_wrapper, test_name, param_kwargs, decorator_fn)
            except Exception as ex:
                # Provides an error message for debugging before rethrowing the exception
                print(
                    f"Failed to instantiate {test_name} for module {optim_info.name}!"
                )
                raise ex

