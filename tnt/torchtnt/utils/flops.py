class FlopTensorDispatchMode(TorchDispatchMode):
    """
    A context manager to measure flops of a module. Requires PyTorch 1.13+.

    Flop count implementation based on
    https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505

    Examples::

        >>> import copy
        >>> import torch
        >>> import torchvision.models as models
        >>> from torchtnt.utils.flops import FlopTensorDispatchMode

        >>> module = models.resnet18()
        >>> module_input = torch.randn(1, 3, 224, 224)
        >>> with FlopTensorDispatchMode(module) as ftdm:
        >>>     # count forward flops
        >>>     res = module(module_input).mean()
        >>>     flops_forward = copy.deepcopy(ftdm.flop_counts)

        >>>     # reset count before counting backward flops
        >>>     ftdm.reset()
        >>>     res.backward()
        >>>     flops_backward = copy.deepcopy(ftdm.flop_counts)

    """

    def __init__(self, module: torch.nn.Module) -> None:
        """
        Initializes a FlopTensorDispatchMode context manager object.

        Args:
            module: The module to count flops on.
        """
        self._all_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._instrument_module(module, "")

        self.flop_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._parents: List[str] = [""]

    # pyre-fixme
    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook_handle in self._all_hooks:
            hook_handle.remove()
        super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(
        self,
        func: Callable[..., Any],  # pyre-fixme [2] func can be any func
        types: Tuple[Any],  # pyre-fixme [2]
        args=(),  # pyre-fixme [2]
        kwargs=None,  # pyre-fixme [2]
    ) -> PyTree:
        rs = func(*args, **kwargs)
        outs = _normalize_tuple(rs)

        if func in flop_mapping:
            flop_count = flop_mapping[func](args, outs)
            for par in self._parents:
                # pyre-fixme [58]
                self.flop_counts[par][func.__name__] += flop_count
        else:
            logging.debug(f"{func} is not yet supported in FLOPs calculation.")

        return rs

    # pyre-fixme [3]
    def _create_backwards_push(self, name: str) -> Callable[..., Any]:
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(
                    lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args
                )
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                parents = self._parents
                parents.append(name)
                return grad_outs

        # Pyre does not support analyzing classes nested in functions.
        # But this class can't be lifted out of the function as it is a static class
        # using a function parameter.
        return PushState.apply

    # pyre-fixme [3]
    def _create_backwards_pop(self, name: str) -> Callable[..., Any]:
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(
                    lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args
                )
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                parents = self._parents
                assert parents[-1] == name
                parents.pop()
                return grad_outs

        # Pyre does not support analyzing classes nested in functions.
        # But this class can't be lifted out of the function as it is a static class
        # using a function parameter.
        return PopState.apply

    # pyre-fixme [3] Return a callable function
    def _enter_module(self, name: str) -> Callable[..., Any]:
        # pyre-fixme [2, 3]
        def f(module: torch.nn.Module, inputs: Tuple[Any]):
            parents = self._parents
            parents.append(name)
            inputs = _normalize_tuple(inputs)
            out = self._create_backwards_pop(name)(*inputs)
            return out

        return f

    # pyre-fixme [3] Return a callable function
    def _exit_module(self, name: str) -> Callable[..., Any]:
        # pyre-fixme [2, 3]
        def f(module: torch.nn.Module, inputs: Tuple[Any], outputs: Tuple[Any]):
            parents = self._parents
            assert parents[-1] == name
            parents.pop()
            outputs = _normalize_tuple(outputs)
            return self._create_backwards_push(name)(*outputs)

        return f

    def _instrument_module(
        self,
        mod: torch.nn.Module,
        par_name: str,
    ) -> None:
        for name, module in dict(mod.named_children()).items():
            formatted_name = name
            if par_name != "":
                formatted_name = f"{par_name}.{name}"
            self._all_hooks.append(
                module.register_forward_pre_hook(self._enter_module(formatted_name))
            )
            self._all_hooks.append(
                module.register_forward_hook(self._exit_module(formatted_name))
            )
            self._instrument_module(module, formatted_name)

    def reset(self) -> None:
        """
        Resets current flop count.
        """
        self._parents = [""]
        self.flop_counts.clear()