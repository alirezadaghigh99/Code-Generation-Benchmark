class MetaTensorMode(object):
    """
    A context manager that enables MetaTensor mode.

    Usage:
        >>> with MetaTensorMode():
        >>>     # all torch.xxx and torch.distributed.xxx will be replaced by patched functions
        >>>     # and the actual execution will be on torch.device('meta')
        >>>     a = torch.rand(100000, 100000)
        >>>     b = torch.rand(100000, 100000)
        >>>     c = torch.mm(a, b)
    """

    def __init__(self):
        self.torch_overrides = {}  # override torch.xxx
        self.dist_overrides = {}  # override torch.distributed.xxx

    def __enter__(self):
        def _dummy(*args, **kwargs):
            pass

        def _new(*args, orig_new=torch.empty, **kwargs):
            return MetaTensor(
                orig_new(*args, **{**kwargs, "device": "meta"}), device=kwargs.get("device", torch.device("cpu"))
            )

        for func in _TorchOverrideableFactoryMethod:
            self.torch_overrides[func] = getattr(torch, func)
            setattr(torch, func, partial(_new, orig_new=getattr(torch, func)))

        for func in _DistCommMethod:
            self.dist_overrides[func] = getattr(dist, func)
            setattr(dist, func, _dummy)

    def __exit__(self, exc_type, exc_value, traceback):
        for func, func_impl in self.torch_overrides.items():
            setattr(torch, func, func_impl)

        for func, func_impl in self.dist_overrides.items():
            setattr(dist, func, func_impl)

