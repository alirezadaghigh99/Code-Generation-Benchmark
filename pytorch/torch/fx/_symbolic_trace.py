def symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
) -> GraphModule:
    """
    Symbolic tracing API

    Given an ``nn.Module`` or function instance ``root``, this function will return a ``GraphModule``
    constructed by recording operations seen while tracing through ``root``.

    ``concrete_args`` allows you to partially specialize your function, whether it's to remove control flow or data structures.

    For example::

        def f(a, b):
            if b == True:
                return a
            else:
                return a*2

    FX can typically not trace through this due to the presence of control
    flow. However, we can use `concrete_args` to specialize on the value of
    `b` to trace through this::

        f = fx.symbolic_trace(f, concrete_args={'b': False})
        assert f(3, False)  == 6

    Note that although you can still pass in different values of `b`, they will be ignored.

    We can also use `concrete_args` to eliminate data-structure handling from
    our function. This will use pytrees to flatten your input. To avoid
    overspecializing, pass in `fx.PH` for values that shouldn't be
    specialized. For example::

        def f(x):
            out = 0
            for v in x.values():
                out += v
            return out
        f = fx.symbolic_trace(f, concrete_args={'x': {'a': fx.PH, 'b': fx.PH, 'c': fx.PH}})
        assert f({'a': 1, 'b': 2, 'c': 4}) == 7


    Args:
        root (Union[torch.nn.Module, Callable]): Module or function to be traced and converted
            into a Graph representation.
        concrete_args (Optional[Dict[str, any]]): Inputs to be partially specialized

    Returns:
        GraphModule: a Module created from the recorded operations from ``root``.
    """
    tracer = Tracer()
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return _make_graph_module(tracer.root, graph, name)