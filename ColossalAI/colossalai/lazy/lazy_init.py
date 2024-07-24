class LazyInitContext:
    """Context manager for lazy initialization. Enables initializing the model without allocating real memory.

    Args:
        tensor_cls (Union[_MyTensor, LazyTensor], optional): This is only for test. Defaults to LazyTensor.
        default_device (Optional[Union[torch.device, str, int]], optional): Defalt device for initialization.
            If it's cuda, initilization will be accelerated, but cuda memory will be allocated. By default, it's cpu.
            Defaults to None.
    """

    _replaced: bool = False

    def __init__(
        self,
        tensor_cls: Union[_MyTensor, LazyTensor] = LazyTensor,
        default_device: Optional[Union[torch.device, str, int]] = None,
    ):
        assert tensor_cls is LazyTensor or tensor_cls is _MyTensor
        self.tensor_cls = tensor_cls
        self.old_default_device = LazyTensor.default_device
        self.default_device = default_device

    def __enter__(self):
        if LazyInitContext._replaced:
            raise RuntimeError(f"LazyInitContext is not reentrant")
        LazyInitContext._replaced = True
        self.old_default_device = self.tensor_cls.default_device
        self.tensor_cls.default_device = self.default_device

        def wrap_factory_method(target):
            # factory functions (eg. torch.empty())
            def wrapper(*args, **kwargs):
                return self.tensor_cls(target, *args, **kwargs)

            return wrapper, target

        def wrap_factory_like_method(orig_target, target):
            # factory_like functions (eg. torch.empty_like())
            def wrapper(*args, **kwargs):
                orig_t = args[0]
                return self.tensor_cls(
                    orig_target, *orig_t.shape, *args[1:], device=orig_t.device, dtype=orig_t.dtype, **kwargs
                )

            return wrapper, target

        def wrap_legacy_constructor(target, dtype):
            # legacy constructor (e.g. torch.LongTensor())
            def wrapper(*args, **kwargs):
                if len(args) == 1 and isinstance(args[0], torch.Tensor):
                    # (Tensor other)
                    return args[0]
                elif len(args) == 1:
                    # (object data, *, torch.device device)
                    kwargs = {**kwargs, "dtype": dtype}
                    replaced, orig = self.overrides["tensor"]
                    return replaced(*args, **kwargs)
                elif _is_int_tuple(args):
                    # (tuple of ints size, *, torch.device device)
                    kwargs = {**kwargs, "dtype": dtype}
                    replaced, orig = self.overrides["empty"]
                    return replaced(*args, **kwargs)
                else:
                    raise TypeError(
                        f"new() received an invalid combination of arguments - got {tuple(type(x) for x in args)}, but expected one of:\n * (Tensor other)\n * (tuple of ints size, *, torch.device device)\n * (object data, *, torch.device device)"
                    )

            return wrapper, target

        def wrap_no_meta_factory(target):
            # factory functions which don't support meta tensor backend
            def wrapper(*args, **kwargs):
                tensor = target(*args, **kwargs)
                return self.tensor_cls(lambda: None, concrete_data=tensor)

            return wrapper, target

        overrides = {
            target: wrap_factory_method(getattr(torch, target))
            for target in _NORMAL_FACTORY
            if callable(getattr(torch, target, None))
        }

        overrides.update(
            {
                target + "_like": wrap_factory_like_method(getattr(torch, target), getattr(torch, target + "_like"))
                for target in _NORMAL_FACTORY
                if callable(getattr(torch, target + "_like", None))
            }
        )

        overrides.update(
            {
                target: wrap_legacy_constructor(getattr(torch, target), dtype)
                for target, dtype in _LEGACY_TENSOR_CONSTRUCTOR.items()
                if callable(getattr(torch, target, None))
            }
        )

        overrides.update(
            {
                target: wrap_no_meta_factory(getattr(torch, target))
                for target in _NO_META_FACTORY
                if callable(getattr(torch, target, None))
            }
        )

        ConstructorManager.apply(overrides)
        PretrainedManager.inject()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tensor_cls.default_device = self.old_default_device
        LazyInitContext._replaced = False
        ConstructorManager.clear()
        PretrainedManager.recover()

    @staticmethod
    def materialize(module: nn.Module, verbose: bool = False) -> nn.Module:
        """Initialize all ``Parameter`` from ``LazyTensor``. This function will modify the module in-place.

        Args:
            module (nn.Module): Target ``nn.Module``
            verbose (bool): Whether to print lazy initialization rate. Defaults to False.
        """

        def apply_fn(name: str, p: LazyTensor):
            p.materialize()

        return _apply_to_lazy_module(module, apply_fn, verbose)

