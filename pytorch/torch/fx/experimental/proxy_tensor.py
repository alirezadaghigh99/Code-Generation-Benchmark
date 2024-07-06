def make_fx(
        f,
        decomposition_table=None,
        tracing_mode="real",
        _allow_non_fake_inputs=False,
        *,
        pre_dispatch=False,
        record_module_stack=False,
        _allow_fake_constant=False,
        _error_on_data_dependent_ops=True):

    assert tracing_mode in ["real", "fake", "symbolic"]


    make_fx_tracer = _MakefxTracer(
        decomposition_table,
        tracing_mode,
        _allow_non_fake_inputs,
        pre_dispatch,
        record_module_stack,
        _allow_fake_constant,
        _error_on_data_dependent_ops
    )

    @functools.wraps(f)
    def wrapped(*args):
        return make_fx_tracer.trace(f, *args)

    return wrapped

