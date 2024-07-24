class modules(_TestParametrizer):
    """ PROTOTYPE: Decorator for specifying a list of modules over which to run a test. """

    def __init__(self, module_info_iterable, allowed_dtypes=None,
                 train_eval_mode=TrainEvalMode.train_and_eval, skip_if_dynamo=True):
        self.module_info_list = list(module_info_iterable)
        self.allowed_dtypes = set(allowed_dtypes) if allowed_dtypes is not None else None
        self.train_eval_mode = train_eval_mode
        self.skip_if_dynamo = skip_if_dynamo

    def _get_training_flags(self, module_info):
        training_flags = []
        if (self.train_eval_mode == TrainEvalMode.train_only or
                self.train_eval_mode == TrainEvalMode.train_and_eval):
            training_flags.append(True)

        if (self.train_eval_mode == TrainEvalMode.eval_only or
                self.train_eval_mode == TrainEvalMode.train_and_eval):
            training_flags.append(False)

        # If train and eval modes don't differ for the module, don't bother using more than one.
        if not module_info.train_and_eval_differ:
            training_flags = training_flags[:1]

        return training_flags

    def _parametrize_test(self, test, generic_cls, device_cls):
        if device_cls is None:
            raise RuntimeError('The @modules decorator is only intended to be used in a device-specific '
                               'context; use it with instantiate_device_type_tests() instead of '
                               'instantiate_parametrized_tests()')

        for module_info in self.module_info_list:
            dtypes = set(module_info.supported_dtypes(device_cls.device_type))
            if self.allowed_dtypes is not None:
                dtypes = dtypes.intersection(self.allowed_dtypes)

            training_flags = self._get_training_flags(module_info)
            for (training, dtype) in product(training_flags, dtypes):
                # Construct the test name; device / dtype parts are handled outside.
                # See [Note: device and dtype suffix placement]
                test_name = module_info.formatted_name
                if len(training_flags) > 1:
                    test_name += f"_{'train_mode' if training else 'eval_mode'}"

                # Construct parameter kwargs to pass to the test.
                param_kwargs = {'module_info': module_info}
                _update_param_kwargs(param_kwargs, 'dtype', dtype)
                _update_param_kwargs(param_kwargs, 'training', training)

                try:

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    if self.skip_if_dynamo and not torch.testing._internal.common_utils.TEST_WITH_TORCHINDUCTOR:
                        test_wrapper = skipIfTorchDynamo("Policy: we don't run ModuleInfo tests w/ Dynamo")(test_wrapper)

                    decorator_fn = partial(module_info.get_decorators, generic_cls.__name__,
                                           test.__name__, device_cls.device_type, dtype)

                    yield (test_wrapper, test_name, param_kwargs, decorator_fn)
                except Exception as ex:
                    # Provides an error message for debugging before rethrowing the exception
                    print(f"Failed to instantiate {test_name} for module {module_info.name}!")
                    raise ex

