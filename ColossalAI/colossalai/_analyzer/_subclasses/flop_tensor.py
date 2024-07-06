def flop_count(module: Union[torch.nn.Module, Callable] = None, *args, verbose: bool = False, **kwargs) -> Number:
    """
    Count the number of floating point operations in a model.
    Ideas from https://pastebin.com/AkvAyJBw.
    Args:
        module (torch.nn.Module): A PyTorch model.
        *args: Input arguments to the model.
        verbose (bool): If True, print the number of flops for each module.
        **kwargs: Input keyword arguments to the model.
    Returns:
        Number: The total number of floating point operations (FWD + BWD).
    """
    maybe_inplace = (
        getattr(module, "inplace", False)
        or kwargs.get("inplace", False)
        or getattr(module, "__name__", None) in ("add_", "mul_", "div_", "sub_")
    )

    class DummyModule(torch.nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func
            self.__name__ = func.__name__

        def forward(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    total_flop_count = {Phase.FWD: 0, Phase.BWD: 0}
    flop_counts = defaultdict(lambda: defaultdict(int))
    parents = ["Global"]
    module = module if isinstance(module, torch.nn.Module) else DummyModule(module)

    class FlopTensor(MetaTensor):
        _tensor: torch.Tensor

        def __repr__(self):
            name = "FlopParameter" if getattr(self, "_is_param", False) else "FlopTensor"
            if self.grad_fn:
                return f"{name}(..., size={tuple(self.shape)}, device='{self.device}', dtype={self.dtype}, grad_fn={self.grad_fn})"
            return f"{name}(..., size={tuple(self.shape)}, device='{self.device}', dtype={self.dtype})"

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            # no_dispatch is only needed if you use enable_python_mode.
            # It prevents infinite recursion.
            rs = super().__torch_dispatch__(func, types, args, kwargs)

            outs = normalize_tuple(rs)

            if func in flop_mapping:
                nonlocal flop_counts, total_flop_count
                flop_count = flop_mapping[func](args, outs)
                for par in parents:
                    flop_counts[par][func.__name__] += flop_count
                total_flop_count[cur_phase] += flop_count

            def wrap(x):
                if isinstance(x, MetaTensor):
                    x = FlopTensor(x)
                return x

            rs = tree_map(wrap, rs)

            return rs

    def is_autogradable(x):
        return isinstance(x, torch.Tensor) and x.is_floating_point()

    def create_backwards_push(name):
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                nonlocal parents
                parents.append(name)
                return grad_outs

        return PushState.apply

    def create_backwards_pop(name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                nonlocal parents
                assert parents[-1] == name
                parents.pop()
                return grad_outs

        return PopState.apply

    def enter_module(name):
        def f(module, inputs):
            nonlocal parents
            parents.append(name)
            inputs = normalize_tuple(inputs)
            out = create_backwards_pop(name)(*inputs)
            return out

        return f

    def exit_module(name):
        def f(module, inputs, outputs):
            nonlocal parents
            assert parents[-1] == name
            parents.pop()
            outputs = normalize_tuple(outputs)
            return create_backwards_push(name)(*outputs)

        return f

    @contextmanager
    def instrument_module(mod):
        registered = []
        for name, module in dict(mod.named_children()).items():
            registered.append(module.register_forward_pre_hook(enter_module(name)))
            registered.append(module.register_forward_hook(exit_module(name)))
        yield
        for handle in registered:
            handle.remove()

    def display_flops():
        for mod in flop_counts.keys():
            print(f"Module: ", mod)
            for k, v in flop_counts[mod].items():
                print("\t", k, _format_flops(v))
            print()

    def detach_variables(r):
        if isinstance(r, torch.Tensor):
            requires_grad = r.requires_grad
            r = r.detach()
            r.requires_grad = requires_grad
        return r

    def wrap(r):
        if isinstance(r, torch.Tensor):
            data_ptr_fn = getattr(r, "_tensor", r).data_ptr
            r = FlopTensor(detach_variables(r))
            if maybe_inplace:
                r = r + 0
            r._tensor.data_ptr = data_ptr_fn
        return r

    with instrument_module(module):
        cur_phase = Phase.FWD
        rst = module(*tree_map(wrap, args), **tree_map(wrap, kwargs))
        rst = tuple(r for r in normalize_tuple(rst) if is_autogradable(r) and r.requires_grad)
        cur_phase = Phase.BWD

        if rst:
            grad = [torch.zeros_like(t) for t in rst]
            torch.autograd.backward(
                rst,
                grad,
            )

    if verbose:
        display_flops()

    return total_flop_count[Phase.FWD], total_flop_count[Phase.BWD]

