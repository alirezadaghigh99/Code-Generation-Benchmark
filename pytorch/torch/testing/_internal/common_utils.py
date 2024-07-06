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

