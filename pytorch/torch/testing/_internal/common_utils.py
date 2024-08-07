def skip_but_pass_in_sandcastle_if(condition, reason):
    """
    Similar to unittest.skipIf, however in the sandcastle environment it just
    "passes" the test instead to avoid creating tasks complaining about tests
    skipping continuously.
    """
    def decorator(func):
        if condition:
            if IS_SANDCASTLE:  # noqa: F821
                @wraps(func)
                def wrapper(*args, **kwargs):
                    print(f'Skipping {func.__name__} on sandcastle for following reason: {reason}', file=sys.stderr)
                return wrapper
            else:
                func.__unittest_skip__ = True
                func.__unittest_skip_why__ = reason

        return func

    return decorator

def skipIfTorchDynamo(msg="test doesn't currently work with dynamo"):
    """
    Usage:
    @skipIfTorchDynamo(msg)
    def test_blah(self):
        ...
    """
    assert isinstance(msg, str), "Are you using skipIfTorchDynamo correctly?"

    def decorator(fn):
        if not isinstance(fn, type):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                if TEST_WITH_TORCHDYNAMO:  # noqa: F821
                    raise unittest.SkipTest(msg)
                else:
                    fn(*args, **kwargs)
            return wrapper

        assert isinstance(fn, type)
        if TEST_WITH_TORCHDYNAMO:  # noqa: F821
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = msg

        return fn

    return decorator

def enable_profiling_mode_for_profiling_tests():
    if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
        old_prof_exec_state = torch._C._jit_set_profiling_executor(True)
        old_prof_mode_state = torch._C._get_graph_executor_optimize(True)
    try:
        yield
    finally:
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            torch._C._jit_set_profiling_executor(old_prof_exec_state)
            torch._C._get_graph_executor_optimize(old_prof_mode_state)

def set_default_dtype(dtype):
    saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(saved_dtype)

def runWithPytorchAPIUsageStderr(code):
        env = os.environ.copy()
        env["PYTORCH_API_USAGE_STDERR"] = "1"
        # remove CI flag since this is a wrapped test process.
        # CI flag should be set in the parent process only.
        if "CI" in env.keys():
            del env["CI"]
        (stdout, stderr) = TestCase.run_process_no_exception(code, env=env)
        return stderr.decode('ascii')

def freeze_rng_state(*args, **kwargs):
    return torch.testing._utils.freeze_rng_state(*args, **kwargs)

def TemporaryFileName(*args, **kwargs):
        # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
        # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
        # close the file after creation and try to remove it manually
        if 'delete' in kwargs:
            if kwargs['delete'] is not False:
                raise UserWarning("only TemporaryFileName with delete=False is supported on Windows.")
        else:
            kwargs['delete'] = False
        f = tempfile.NamedTemporaryFile(*args, **kwargs)
        try:
            f.close()
            yield f.name
        finally:
            os.unlink(f.name)

def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if TEST_NUMPY:
        np.random.seed(seed)

def gradcheck(fn, inputs, **kwargs):
    # Wrapper around gradcheck that enables certain keys by default.
    # Use this testing-internal gradcheck instead of autograd.gradcheck so that new features like vmap and
    # forward-mode AD are tested by default. We create this wrapper because we'd like to keep new checks
    # to be disabled to default for the public-facing api to avoid breaking user code.
    #
    # All PyTorch devs doing testing should use this wrapper instead of autograd.gradcheck.
    default_values = {
        "check_batched_grad": True,
        "fast_mode": True,
    }

    if TEST_WITH_SLOW_GRADCHECK:  # noqa: F821
        default_values["fast_mode"] = False

    for key, value in default_values.items():
        # default value override values explicitly set to None
        k = kwargs.get(key, None)
        kwargs[key] = k if k is not None else value

    return torch.autograd.gradcheck(fn, inputs, **kwargs)

def serialTest(condition=True):
    """
    Decorator for running tests serially.  Requires pytest
    """
    def decorator(fn):
        if has_pytest and condition:
            return pytest.mark.serial(fn)
        return fn
    return decorator

def skipIfTorchInductor(msg="test doesn't currently work with torchinductor",
                        condition=TEST_WITH_TORCHINDUCTOR):  # noqa: F821
    def decorator(fn):
        if not isinstance(fn, type):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                if condition:
                    raise unittest.SkipTest(msg)
                else:
                    fn(*args, **kwargs)
            return wrapper

        assert isinstance(fn, type)
        if condition:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = msg

        return fn

    return decorator

def first_sample(self: unittest.TestCase, samples: Iterable[T]) -> T:
    """
    Returns the first sample from an iterable of samples, like those returned by OpInfo.
    The test will be skipped if no samples are available.
    """
    try:
        return next(iter(samples))
    except StopIteration as e:
        raise unittest.SkipTest('Skipped! Need at least 1 sample input') from e

def clone_input_helper(input):
    if isinstance(input, torch.Tensor):
        return torch.clone(input)

    if isinstance(input, Sequence):
        return tuple(map(clone_input_helper, input))

    return input

class parametrize(_TestParametrizer):
    """
    Decorator for applying generic test parametrizations.

    The interface for this decorator is modeled after `@pytest.mark.parametrize`.
    Basic usage between this decorator and pytest's is identical. The first argument
    should be a string containing comma-separated names of parameters for the test, and
    the second argument should be an iterable returning values or tuples of values for
    the case of multiple parameters.

    Beyond this basic usage, the decorator provides some additional functionality that
    pytest does not.

    1. Parametrized tests end up as generated test functions on unittest test classes.
    Since this differs from how pytest works, this decorator takes on the additional
    responsibility of naming these test functions. The default test names consists of
    the test's base name followed by each parameter name + value (e.g. "test_bar_x_1_y_foo"),
    but custom names can be defined using `name_fn` or the `subtest` structure (see below).

    2. The decorator specially handles parameter values of type `subtest`, which allows for
    more fine-grained control over both test naming and test execution. In particular, it can
    be used to tag subtests with explicit test names or apply arbitrary decorators (see examples
    below).

    Examples::

        @parametrize("x", range(5))
        def test_foo(self, x):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')])
        def test_bar(self, x, y):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')],
                     name_fn=lambda x, y: '{}_{}'.format(x, y))
        def test_bar_custom_names(self, x, y):
            ...

        @parametrize("x, y", [subtest((1, 2), name='double'),
                              subtest((1, 3), name='triple', decorators=[unittest.expectedFailure]),
                              subtest((1, 4), name='quadruple')])
        def test_baz(self, x, y):
            ...

    To actually instantiate the parametrized tests, one of instantiate_parametrized_tests() or
    instantiate_device_type_tests() should be called. The former is intended for test classes
    that contain device-agnostic tests, while the latter should be used for test classes that
    contain device-specific tests. Both support arbitrary parametrizations using the decorator.

    Args:
        arg_str (str): String of arg names separate by commas (e.g. "x,y").
        arg_values (iterable): Iterable of arg values (e.g. range(10)) or
            tuples of arg values (e.g. [(1, 2), (3, 4)]).
        name_fn (Callable): Optional function that takes in parameters and returns subtest name.
    """
    def __init__(self, arg_str, arg_values, name_fn=None):
        self.arg_names: List[str] = [s.strip() for s in arg_str.split(',') if s != '']
        self.arg_values = arg_values
        self.name_fn = name_fn

    def _formatted_str_repr(self, idx, name, value):
        """ Returns a string representation for the given arg that is suitable for use in test function names. """
        if isinstance(value, torch.dtype):
            return dtype_name(value)
        elif isinstance(value, torch.device):
            return str(value)
        # Can't use isinstance as it would cause a circular import
        elif type(value).__name__ in {'OpInfo', 'ModuleInfo'}:
            return value.formatted_name
        elif isinstance(value, (int, float, str)):
            return f"{name}_{str(value).replace('.', '_')}"
        else:
            return f"{name}{idx}"

    def _default_subtest_name(self, idx, values):
        return '_'.join([self._formatted_str_repr(idx, a, v) for a, v in zip(self.arg_names, values)])

    def _get_subtest_name(self, idx, values, explicit_name=None):
        if explicit_name:
            subtest_name = explicit_name
        elif self.name_fn:
            subtest_name = self.name_fn(*values)
        else:
            subtest_name = self._default_subtest_name(idx, values)
        return subtest_name

    def _parametrize_test(self, test, generic_cls, device_cls):
        if len(self.arg_names) == 0:
            # No additional parameters needed for the test.
            test_name = ''
            yield (test, test_name, {}, lambda _: [])
        else:
            # Each "values" item is expected to be either:
            # * A tuple of values with one for each arg. For a single arg, a single item is expected.
            # * A subtest instance with arg_values matching the previous.
            values = check_exhausted_iterator = object()
            for idx, values in enumerate(self.arg_values):
                maybe_name = None

                decorators = []
                if isinstance(values, subtest):
                    sub = values
                    values = sub.arg_values
                    maybe_name = sub.name

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    decorators = sub.decorators
                    gen_test = test_wrapper
                else:
                    gen_test = test

                values = list(values) if len(self.arg_names) > 1 else [values]
                if len(values) != len(self.arg_names):
                    raise RuntimeError(f'Expected # values == # arg names, but got: {len(values)} '
                                       f'values and {len(self.arg_names)} names for test "{test.__name__}"')

                param_kwargs = dict(zip(self.arg_names, values))

                test_name = self._get_subtest_name(idx, values, explicit_name=maybe_name)

                def decorator_fn(_, decorators=decorators):
                    return decorators

                yield (gen_test, test_name, param_kwargs, decorator_fn)

            if values is check_exhausted_iterator:
                raise ValueError(f'{test}: An empty arg_values was passed to @parametrize. '
                                 'Note that this may result from reuse of a generator.')

class DeterministicGuard:
    def __init__(self, deterministic, *, warn_only=False, fill_uninitialized_memory=True):
        self.deterministic = deterministic
        self.warn_only = warn_only
        self.fill_uninitialized_memory = fill_uninitialized_memory

    def __enter__(self):
        self.deterministic_restore = torch.are_deterministic_algorithms_enabled()
        self.warn_only_restore = torch.is_deterministic_algorithms_warn_only_enabled()
        self.fill_uninitialized_memory_restore = torch.utils.deterministic.fill_uninitialized_memory
        torch.use_deterministic_algorithms(
            self.deterministic,
            warn_only=self.warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = self.fill_uninitialized_memory

    def __exit__(self, exception_type, exception_value, traceback):
        torch.use_deterministic_algorithms(
            self.deterministic_restore,
            warn_only=self.warn_only_restore)
        torch.utils.deterministic.fill_uninitialized_memory = self.fill_uninitialized_memory_restore

class CudaSyncGuard:
    def __init__(self, sync_debug_mode):
        self.mode = sync_debug_mode

    def __enter__(self):
        self.debug_mode_restore = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode(self.mode)

    def __exit__(self, exception_type, exception_value, traceback):
        torch.cuda.set_sync_debug_mode(self.debug_mode_restore)

class subtest:
    """
    Explicit subtest case for use with test parametrization.
    Allows for explicit naming of individual subtest cases as well as applying
    decorators to the parametrized test.

    Args:
        arg_values (iterable): Iterable of arg values (e.g. range(10)) or
            tuples of arg values (e.g. [(1, 2), (3, 4)]).
        name (str): Optional name to use for the test.
        decorators (iterable): Iterable of decorators to apply to the generated test.
    """
    __slots__ = ['arg_values', 'name', 'decorators']

    def __init__(self, arg_values, name=None, decorators=None):
        self.arg_values = arg_values
        self.name = name
        self.decorators = decorators if decorators else []

class swap(_TestParametrizer):
    def __init__(self, swap_values):
        super().__init__()
        self.swap_values = swap_values

    def _parametrize_test(self, test, generic_cls, device_cls):
        for swap in self.swap_values:
            yield wrapSwapTensorsTest(swap)(test), f'swap_{swap}', {}, lambda _: []

class decorateIf(_TestParametrizer):
    """
    Decorator for applying parameter-specific conditional decoration.
    Composes with other test parametrizers (e.g. @modules, @ops, @parametrize, etc.).

    Examples::

        @decorateIf(unittest.skip, lambda params: params["x"] == 2)
        @parametrize("x", range(5))
        def test_foo(self, x):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')])
        @decorateIf(
            unittest.expectedFailure,
            lambda params: params["x"] == 3 and params["y"] == "baz"
        )
        def test_bar(self, x, y):
            ...

        @decorateIf(
            unittest.expectedFailure,
            lambda params: params["op"].name == "add" and params["dtype"] == torch.float16
        )
        @ops(op_db)
        def test_op_foo(self, device, dtype, op):
            ...

        @decorateIf(
            unittest.skip,
            lambda params: params["module_info"].module_cls is torch.nn.Linear and \
                params["device"] == "cpu"
        )
        @modules(module_db)
        def test_module_foo(self, device, dtype, module_info):
            ...

    Args:
        decorator: Test decorator to apply if the predicate is satisfied.
        predicate_fn (Callable): Function taking in a dict of params and returning a boolean
            indicating whether the decorator should be applied or not.
    """
    def __init__(self, decorator, predicate_fn):
        self.decorator = decorator
        self.predicate_fn = predicate_fn

    def _parametrize_test(self, test, generic_cls, device_cls):

        # Leave test as-is and return the appropriate decorator_fn.
        def decorator_fn(params, decorator=self.decorator, predicate_fn=self.predicate_fn):
            if predicate_fn(params):
                return [decorator]
            else:
                return []

        @wraps(test)
        def test_wrapper(*args, **kwargs):
            return test(*args, **kwargs)

        test_name = ''
        yield (test_wrapper, test_name, {}, decorator_fn)

class BytesIOContext(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

