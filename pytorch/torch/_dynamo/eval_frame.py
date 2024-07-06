def optimize(*args, **kwargs):
    def rebuild_ctx():
        return optimize(*args, **kwargs)

    return _optimize(rebuild_ctx, *args, **kwargs)

def optimize_assert(
    backend,
    *,
    hooks=Hooks(None, None),
    export=False,
    export_constraints=None,
    dynamic=None,
    rebuild_ctx=None,
):
    """
    The same as `torch._dynamo.optimize(backend, nopython=True)`
    """
    backend = get_compiler_fn(backend)

    # Find if backend has any extra context manager
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    return _optimize_catch_errors(
        convert_frame.convert_frame_assert(
            backend, export=export, export_constraints=export_constraints
        ),
        hooks,
        backend_ctx_ctor,
        export=export,
        dynamic=dynamic,
        rebuild_ctx=rebuild_ctx,
    )

