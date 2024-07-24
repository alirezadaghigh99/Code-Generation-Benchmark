class HigherOrderOperator(OperatorBase):
    # The HigherOrderOperator will appear as torch.ops.higher_order.{name}
    #
    # If you're creating a new HigherOrderOperator, please do not change the
    # default. Adding operators to the global torch.ops namespace is a bad
    # practice due to name collisions.
    def __init__(self, name):
        super().__init__()
        self._name = name

        # Make _OPNamespace not scream, this whole name based association needs a good hard look
        self.__name__ = name
        _higher_order_ops[name] = self
        self._ns = "higher_order"

        # For a normal HigherOrderOperator instance, we will change its __module__ from torch._ops to
        # torch._ops.higher_order.
        # For an instance of subclass of HigherOrderOperator (e.g. customized higher order op),
        # the __module__ attribute will be kept unchanged.
        if self.__class__ is HigherOrderOperator:
            self_name_space = "." + self.namespace if self.namespace else ""
            self.__module__ = self.__module__ + self_name_space

        self.non_fallthrough_keys = torch._C._dispatch_keyset_full()

        for dispatch_key in _HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS:
            self.fallthrough(dispatch_key)

        # [NOTE] We have to register pre-dispatch key implementation
        # because sometimes HOP use aot-dispatch tracing to detect certaion
        # mutations. This is problematic when we are functionalizing HOP
        # during pre-dispatch because when the inner tracer starts, it will see
        # that PreDispatch key is still active. In that case, we just redispatch
        # it to next key. This is only safe to do when PreDispatch key stack has no
        # active modes.

    def py_impl(self, k):
        if isinstance(k, torch._C.DispatchKey) and not self.non_fallthrough_keys.has(k):
            self.non_fallthrough_keys = self.non_fallthrough_keys.add(k)
        return super().py_impl(k)

    @property
    def namespace(self):
        return self._ns

    def fallthrough(self, dispatch_key):
        self.non_fallthrough_keys = self.non_fallthrough_keys.remove(dispatch_key)

    def dispatch(self, dispatch_key, *args, **kwargs):
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        if dispatch_key in self._dispatch_cache:
            kernel = self._dispatch_cache[dispatch_key]
            assert not isinstance(kernel, torch._C.DispatchKey)
            return kernel(*args, **kwargs)

        if dispatch_key == torch._C.DispatchKey.FuncTorchDynamicLayerFrontMode:
            return dispatch_functorch(self, args, kwargs)

        if dispatch_key == torch._C.DispatchKey.Python:
            # The place to handle ProxyTorchDispatchMode, FakeTensorMode, etc
            from torch.utils._python_dispatch import _pop_mode_temporarily

            curr_mode = _get_current_dispatch_mode()
            assert (
                curr_mode is not None
            ), "Illegal invocation of dispatch on torch._C.DispatchKey.Python without a mode."
            assert (
                type(curr_mode) in self.python_key_mode_table
            ), f"Current active mode {curr_mode} not registered"
            handler = self.python_key_mode_table[type(curr_mode)]
            with _pop_mode_temporarily() as mode:
                return handler(mode, *args, **kwargs)

        functionality_key = torch._C._to_functionality_key(dispatch_key)  # type: ignore[attr-defined]
        if functionality_key == torch._C.DispatchKey.PreDispatch:
            from torch.utils._python_dispatch import _pop_mode_temporarily

            # The check for Python in the exclude set is so we properly respect `with no_dispatch()`
            # calls inside of a mode.
            if (
                _len_torch_dispatch_stack_pre_dispatch() > 0
            ) and not torch._C._dispatch_tls_is_dispatch_key_excluded(
                DispatchKey.Python
            ):
                curr_mode = _get_current_dispatch_mode_pre_dispatch()
                assert (
                    curr_mode is not None
                ), "Illegal invocation of dispatch on torch._C.DispatchKey.PreDispatch without a mode."
                assert (
                    type(curr_mode) in self.python_key_mode_table
                ), f"Current active mode {curr_mode} not registered"
                handler = self.python_key_mode_table[type(curr_mode)]
                with _pop_mode_temporarily(functionality_key) as mode:
                    return handler(mode, *args, **kwargs)

        final_key = resolve_key(self, dispatch_key)

        # This can current fail due to backend fallbacks.  You just have to
        # register them by hand for HigherOrderOperator.
        if final_key not in self.py_kernels:
            raise NotImplementedError(
                f"could not find kernel for HigherOrderOperator {self._name} "
                f"at dispatch key {final_key} (resolved from {dispatch_key})"
            )

        # [NOTE] We shouldn't cache PreDispatch kernel here because depending
        # on what modes are active, predispatch behaviour is different.
        # Also we do same thing for normal ops:
        # See Note [Not Caching Per-Dispatch-Key Mode Handlers]
        if dispatch_key != torch._C.DispatchKey.PreDispatch:
            self._dispatch_cache[dispatch_key] = self.py_kernels[final_key]
        kernel = self.py_kernels[final_key]
        # It's illegal to register DispatchKey to py_kernels, since there's no
        # C++ kernel to call into
        assert not isinstance(kernel, torch._C.DispatchKey)
        return kernel(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        # Dynamo already traces the body of HigherOrderOp beforehand when it
        # so no need to trace into it.
        import torch._dynamo
        from torch._dynamo import disable

        @disable
        def wrapper():
            flat_args = _to_flat_tuple(args, kwargs)
            if torch.overrides.has_torch_function(flat_args):
                return torch.overrides.handle_torch_function(
                    self, flat_args, *args, **kwargs
                )

            dispatch_key_set = _compute_keyset(args, kwargs, self.non_fallthrough_keys)
            return self.dispatch(
                dispatch_key_set.highestPriorityTypeId(), *args, **kwargs
            )

        return wrapper()

    def __str__(self):
        return f"{self.name()}"

    # def __repr__(self):
    #     return f"torch.ops._higher_order_ops.{self._name}"

    def name(self):
        return self._name

