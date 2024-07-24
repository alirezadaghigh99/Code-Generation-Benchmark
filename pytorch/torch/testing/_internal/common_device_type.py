def _has_sufficient_memory(device, size):
    if torch.device(device).type == "cuda":
        if not torch.cuda.is_available():
            return False
        gc.collect()
        torch.cuda.empty_cache()
        # torch.cuda.mem_get_info, aka cudaMemGetInfo, returns a tuple of (free memory, total memory) of a GPU
        if device == "cuda":
            device = "cuda:0"
        return torch.cuda.memory.mem_get_info(device)[0] >= size

    if device == "xla":
        raise unittest.SkipTest("TODO: Memory availability checks for XLA?")

    if device == "xpu":
        raise unittest.SkipTest("TODO: Memory availability checks for Intel GPU?")

    if device != "cpu":
        raise unittest.SkipTest("Unknown device type")

    # CPU
    if not HAS_PSUTIL:
        raise unittest.SkipTest("Need psutil to determine if memory is sufficient")

    # The sanitizers have significant memory overheads
    if TEST_WITH_ASAN or TEST_WITH_TSAN or TEST_WITH_UBSAN:
        effective_size = size * 10
    else:
        effective_size = size

    if psutil.virtual_memory().available < effective_size:
        gc.collect()
    return psutil.virtual_memory().available >= effective_size

def get_all_device_types() -> List[str]:
    return ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]

class dtypes:
    def __init__(self, *args, device_type="all"):
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            for arg in args:
                assert isinstance(arg, (list, tuple)), (
                    "When one dtype variant is a tuple or list, "
                    "all dtype variants must be. "
                    f"Received non-list non-tuple dtype {str(arg)}"
                )
                assert all(
                    isinstance(dtype, torch.dtype) for dtype in arg
                ), f"Unknown dtype in {str(arg)}"
        else:
            assert all(
                isinstance(arg, torch.dtype) for arg in args
            ), f"Unknown dtype in {str(args)}"

        self.args = args
        self.device_type = device_type

    def __call__(self, fn):
        d = getattr(fn, "dtypes", {})
        assert self.device_type not in d, f"dtypes redefinition for {self.device_type}"
        d[self.device_type] = self.args
        fn.dtypes = d
        return fn

class ops(_TestParametrizer):
    def __init__(
        self,
        op_list,
        *,
        dtypes: Union[OpDTypes, Sequence[torch.dtype]] = OpDTypes.supported,
        allowed_dtypes: Optional[Sequence[torch.dtype]] = None,
        skip_if_dynamo=True,
    ):
        self.op_list = list(op_list)
        self.opinfo_dtypes = dtypes
        self.allowed_dtypes = (
            set(allowed_dtypes) if allowed_dtypes is not None else None
        )
        self.skip_if_dynamo = skip_if_dynamo

    def _parametrize_test(self, test, generic_cls, device_cls):
        """Parameterizes the given test function across each op and its associated dtypes."""
        if device_cls is None:
            raise RuntimeError(
                "The @ops decorator is only intended to be used in a device-specific "
                "context; use it with instantiate_device_type_tests() instead of "
                "instantiate_parametrized_tests()"
            )

        op = check_exhausted_iterator = object()
        for op in self.op_list:
            # Determine the set of dtypes to use.
            dtypes: Union[Set[torch.dtype], Set[None]]
            if isinstance(self.opinfo_dtypes, Sequence):
                dtypes = set(self.opinfo_dtypes)
            elif self.opinfo_dtypes == OpDTypes.unsupported_backward:
                dtypes = set(get_all_dtypes()).difference(
                    op.supported_backward_dtypes(device_cls.device_type)
                )
            elif self.opinfo_dtypes == OpDTypes.supported_backward:
                dtypes = op.supported_backward_dtypes(device_cls.device_type)
            elif self.opinfo_dtypes == OpDTypes.unsupported:
                dtypes = set(get_all_dtypes()).difference(
                    op.supported_dtypes(device_cls.device_type)
                )
            elif self.opinfo_dtypes == OpDTypes.supported:
                dtypes = set(op.supported_dtypes(device_cls.device_type))
            elif self.opinfo_dtypes == OpDTypes.any_one:
                # Tries to pick a dtype that supports both forward or backward
                supported = op.supported_dtypes(device_cls.device_type)
                supported_backward = op.supported_backward_dtypes(
                    device_cls.device_type
                )
                supported_both = supported.intersection(supported_backward)
                dtype_set = supported_both if len(supported_both) > 0 else supported
                for dtype in ANY_DTYPE_ORDER:
                    if dtype in dtype_set:
                        dtypes = {dtype}
                        break
                else:
                    dtypes = {}
            elif self.opinfo_dtypes == OpDTypes.any_common_cpu_cuda_one:
                # Tries to pick a dtype that supports both CPU and CUDA
                supported = set(op.dtypes).intersection(op.dtypesIfCUDA)
                if supported:
                    dtypes = {
                        next(dtype for dtype in ANY_DTYPE_ORDER if dtype in supported)
                    }
                else:
                    dtypes = {}

            elif self.opinfo_dtypes == OpDTypes.none:
                dtypes = {None}
            else:
                raise RuntimeError(f"Unknown OpDType: {self.opinfo_dtypes}")

            if self.allowed_dtypes is not None:
                dtypes = dtypes.intersection(self.allowed_dtypes)

            # Construct the test name; device / dtype parts are handled outside.
            # See [Note: device and dtype suffix placement]
            test_name = op.formatted_name

            for dtype in dtypes:
                # Construct parameter kwargs to pass to the test.
                param_kwargs = {"op": op}
                _update_param_kwargs(param_kwargs, "dtype", dtype)

                # NOTE: test_wrapper exists because we don't want to apply
                #   op-specific decorators to the original test.
                #   Test-specific decorators are applied to the original test,
                #   however.
                try:

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        try:
                            return test(*args, **kwargs)
                        except unittest.SkipTest as e:
                            raise e
                        except Exception as e:
                            tracked_input = get_tracked_input()
                            if PRINT_REPRO_ON_FAILURE and tracked_input is not None:
                                raise Exception(  # noqa: TRY002
                                    f"Caused by {tracked_input.type_desc} "
                                    f"at index {tracked_input.index}: "
                                    f"{_serialize_sample(tracked_input.val)}"
                                ) from e
                            raise e
                        finally:
                            clear_tracked_input()

                    if self.skip_if_dynamo and not TEST_WITH_TORCHINDUCTOR:
                        test_wrapper = skipIfTorchDynamo(
                            "Policy: we don't run OpInfo tests w/ Dynamo"
                        )(test_wrapper)

                    # Initialize info for the last input seen. This is useful for tracking
                    # down which inputs caused a test failure. Note that TrackedInputIter is
                    # responsible for managing this.
                    test.tracked_input = None

                    decorator_fn = partial(
                        op.get_decorators,
                        generic_cls.__name__,
                        test.__name__,
                        device_cls.device_type,
                        dtype,
                    )

                    yield (test_wrapper, test_name, param_kwargs, decorator_fn)
                except Exception as ex:
                    # Provides an error message for debugging before rethrowing the exception
                    print(f"Failed to instantiate {test_name} for op {op.name}!")
                    raise ex
        if op is check_exhausted_iterator:
            raise ValueError(
                "An empty op_list was passed to @ops. "
                "Note that this may result from reuse of a generator."
            )

class deviceCountAtLeast:
    def __init__(self, num_required_devices):
        self.num_required_devices = num_required_devices

    def __call__(self, fn):
        assert not hasattr(
            fn, "num_required_devices"
        ), f"deviceCountAtLeast redefinition for {fn.__name__}"
        fn.num_required_devices = self.num_required_devices

        @wraps(fn)
        def multi_fn(slf, devices, *args, **kwargs):
            if len(devices) < self.num_required_devices:
                reason = f"fewer than {self.num_required_devices} devices detected"
                raise unittest.SkipTest(reason)

            return fn(slf, devices, *args, **kwargs)

        return multi_fn

class dtypesIfCUDA(dtypes):
    def __init__(self, *args):
        super().__init__(*args, device_type="cuda")

class precisionOverride:
    def __init__(self, d):
        assert isinstance(
            d, dict
        ), "precisionOverride not given a dtype : precision dict!"
        for dtype in d.keys():
            assert isinstance(
                dtype, torch.dtype
            ), f"precisionOverride given unknown dtype {dtype}"

        self.d = d

    def __call__(self, fn):
        fn.precision_overrides = self.d
        return fn

class dtypesIfCPU(dtypes):
    def __init__(self, *args):
        super().__init__(*args, device_type="cpu")

class toleranceOverride:
    def __init__(self, d):
        assert isinstance(d, dict), "toleranceOverride not given a dtype : tol dict!"
        for dtype, prec in d.items():
            assert isinstance(
                dtype, torch.dtype
            ), f"toleranceOverride given unknown dtype {dtype}"
            assert isinstance(
                prec, tol
            ), "toleranceOverride not given a dtype : tol dict!"

        self.d = d

    def __call__(self, fn):
        fn.tolerance_overrides = self.d
        return fn

class skipCPUIf(skipIf):
    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type="cpu")

