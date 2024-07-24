class Proxy:
    """
    ``Proxy`` objects are ``Node`` wrappers that flow through the
    program during symbolic tracing and record all the operations
    (``torch`` function calls, method calls, operators) that they touch
    into the growing FX Graph.

    If you're doing graph transforms, you can wrap your own ``Proxy``
    method around a raw ``Node`` so that you can use the overloaded
    operators to add additional things to a ``Graph``.

    ``Proxy`` objects cannot be iterated. In other words, the symbolic
    tracer will throw an error if a ``Proxy`` is used in a loop or as
    an ``*args``/``**kwargs`` function argument.

    There are two main ways around this:
    1. Factor out the untraceable logic into a top-level function and
    use ``fx.wrap`` on it.
    2. If the control flow is static (i.e. the loop trip count is
    based on some hyperparameter), the code can be kept in its original
    position and refactored into something like::

        for i in range(self.some_hyperparameter):
            indexed_item = proxied_value[i]

    For a more detailed description into the Proxy internals, check out
    the "Proxy" section in `torch/fx/README.md`
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, node: Node, tracer: 'Optional[TracerBase]' = None):
        if tracer is None:
            # This allows you to create a Proxy object around a raw Node
            tracer = GraphAppendingTracer(node.graph)
        self.tracer = tracer
        self.node = node

    def __repr__(self) -> str:
        return f'Proxy({self.node.name})'

    def __getattr__(self, k) -> 'Attribute':
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return Attribute(self, k)

    def __call__(self, *args, **kwargs) -> 'Proxy':
        return self.tracer.create_proxy('call_method', '__call__', (self,) + args, kwargs)

    def __iter__(self) -> Iterator['Proxy']:
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        inst_list = list(dis.get_instructions(calling_frame.f_code))
        if sys.version_info >= (3, 11):
            from bisect import bisect_left
            inst_idx = bisect_left(inst_list, calling_frame.f_lasti, key=lambda x: x.offset)
        else:
            inst_idx = calling_frame.f_lasti // 2
        inst = inst_list[inst_idx]
        if inst.opname == 'UNPACK_SEQUENCE':
            return (self[i] for i in range(inst.argval))  # type: ignore[index]

        return self.tracer.iter(self)

    def __abs__(self):
        return self.tracer.create_proxy('call_function', operator.abs, (self,), {})

    def __bool__(self) -> bool:
        if self.tracer.trace_asserts:
            # check if this boolean is used in an assertion, bytecode pattern for assertions
            # is pretty stable for Python 3.7--3.9
            frame = inspect.currentframe()
            assert frame is not None
            calling_frame = frame.f_back
            assert calling_frame is not None
            insts = list(dis.get_instructions(calling_frame.f_code))
            if sys.version_info >= (3, 11):
                from bisect import bisect_left
                cur = bisect_left(insts, calling_frame.f_lasti, key=lambda x: x.offset)
            else:
                cur = calling_frame.f_lasti // 2
            inst = insts[cur]

            if inst.opname == 'POP_JUMP_IF_TRUE':
                first = insts[cur + 1]
                assert inst.arg is not None
                last = insts[inst.arg // 2 - 1]
                starts_with_assert = (first.opname == 'LOAD_GLOBAL' and first.argval == 'AssertionError'
                                      or first.opname == 'LOAD_ASSERTION_ERROR')
                if starts_with_assert and last.opname == 'RAISE_VARARGS':
                    self.tracer.create_proxy('call_function', assert_fn, (self,), {})
                    return True

        return self.tracer.to_bool(self)

    @compatibility(is_backward_compatible=True)
    def keys(self):
        return self.tracer.keys(self)

    def __len__(self):
        raise RuntimeError("'len' is not supported in symbolic tracing by default. If you want "
                           "this call to be recorded, please call torch.fx.wrap('len') at "
                           "module scope")

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        tracers : Dict[Any, None] = {}

        def find_tracer(a):
            if isinstance(a, cls):
                tracers[a.tracer] = None
        torch.fx.node.map_aggregate(args, find_tracer)
        torch.fx.node.map_aggregate(kwargs, find_tracer)

        if len(tracers) > 1:
            raise RuntimeError(f'Found multiple different tracers {list(tracers.keys())} while '
                               f'trying to trace operations {orig_method}')
        tracer = next(iter(tracers.keys()))

        if isinstance(orig_method, torch._C.ScriptMethod):
            args = (orig_method.owner,) + args
            return tracer.create_proxy('call_method', orig_method.name, args, kwargs)
        if torch.overrides.is_tensor_method_or_property(orig_method):
            return tracer.create_proxy('call_method', orig_method.__name__, args, kwargs)
        else:
            if isinstance(orig_method, torch._ops.HigherOrderOperator):
                # TODO: Define how to symbolically trace HigherOrderOperators
                raise RuntimeError("Unable to symbolically trace HigherOrderOperators")
            return tracer.create_proxy('call_function', orig_method, args, kwargs,
                                       name=tracer.graph._target_to_str(orig_method.__name__))

