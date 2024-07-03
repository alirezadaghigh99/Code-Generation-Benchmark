class CompileCounter:
    def __init__(self):
        self.frame_count = 0
        self.op_count = 0

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        return gm.forward

    def clear(self):
        self.frame_count = 0
        self.op_count = 0class EagerAndRecordGraphs:
    def __init__(self):
        self.graphs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        self.graphs.append(gm)
        return gm.forwardclass CompileCounterWithBackend:
    def __init__(self, backend):
        self.frame_count = 0
        self.op_count = 0
        self.backend = backend
        self.graphs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        from .backends.registry import lookup_backend

        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        self.graphs.append(gm)
        return lookup_backend(self.backend)(gm, example_inputs)def rand_strided(
    size: Sequence[int],
    stride: Sequence[int],
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
    extra_size: int = 0,
):
    needed_size = (
        sum((shape - 1) * stride for shape, stride in zip(size, stride))
        + 1
        + extra_size
    )
    if dtype.is_floating_point:
        if dtype.itemsize == 1:
            """
            normal distribution kernel is not implemented for fp8..
            Workaround that by creating a fp16 tensor and then cast.
            """
            buffer = torch.randn(needed_size, dtype=torch.float16, device=device).to(
                dtype=dtype
            )
        else:
            buffer = torch.randn(needed_size, dtype=dtype, device=device)
    else:
        buffer = torch.zeros(size=[needed_size], dtype=dtype, device=device)
    return torch.as_strided(buffer, size, stride)def make_test_cls_with_patches(
    cls, cls_prefix, fn_suffix, *patches, xfail_prop=None, decorator=lambda x: x
):
    DummyTestClass = type(f"{cls_prefix}{cls.__name__}", cls.__bases__, {})
    DummyTestClass.__qualname__ = DummyTestClass.__name__

    for name in dir(cls):
        if name.startswith("test_"):
            fn = getattr(cls, name)
            if not callable(fn):
                setattr(DummyTestClass, name, getattr(cls, name))
                continue
            new_name = f"{name}{fn_suffix}"
            new_fn = _make_fn_with_patches(fn, *patches)
            new_fn.__name__ = new_name
            if xfail_prop is not None and hasattr(fn, xfail_prop):
                new_fn = unittest.expectedFailure(new_fn)
            setattr(DummyTestClass, new_name, decorator(new_fn))
        # NB: Doesn't handle slots correctly, but whatever
        elif not hasattr(DummyTestClass, name):
            setattr(DummyTestClass, name, getattr(cls, name))

    return DummyTestClassclass CompileCounter:
    def __init__(self):
        self.frame_count = 0
        self.op_count = 0

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        return gm.forward

    def clear(self):
        self.frame_count = 0
        self.op_count = 0class EagerAndRecordGraphs:
    def __init__(self):
        self.graphs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        self.graphs.append(gm)
        return gm.forwardclass CompileCounterWithBackend:
    def __init__(self, backend):
        self.frame_count = 0
        self.op_count = 0
        self.backend = backend
        self.graphs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        from .backends.registry import lookup_backend

        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        self.graphs.append(gm)
        return lookup_backend(self.backend)(gm, example_inputs)