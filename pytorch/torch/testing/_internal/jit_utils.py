def make_global(*args):
    for arg in args:
        setattr(sys.modules[arg.__module__], arg.__name__, arg)

def _get_py3_code(code, fn_name):
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = os.path.join(tmp_dir, 'script.py')
        with open(script_path, 'w') as f:
            f.write(code)
        spec = importlib.util.spec_from_file_location(fn_name, script_path)
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        assert isinstance(loader, Loader)  # Assert type to meet MyPy requirement
        loader.exec_module(module)
        fn = getattr(module, fn_name)
        return fn

def set_fusion_group_inlining(inlining):
    old = torch._C._debug_get_fusion_group_inlining()
    torch._C._debug_set_fusion_group_inlining(inlining)
    try:
        yield
    finally:
        torch._C._debug_set_fusion_group_inlining(old)

def get_execution_plan(graph_executor_state):
    execution_plans = list(graph_executor_state.execution_plans.values())
    num_plans = len(execution_plans)
    if num_plans != 1:
        raise RuntimeError('This test assumes this GraphExecutor should '
                           f'only have one execution plan, got: {num_plans}')
    return execution_plans[0]

