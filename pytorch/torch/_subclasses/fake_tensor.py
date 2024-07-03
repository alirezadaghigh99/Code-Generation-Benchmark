class FakeTensorMode(TorchDispatchMode):
    cache: Dict[_DispatchCacheKey, _DispatchCacheEntry] = {}
    cache_hits: int = 0
    cache_misses: int = 0
    cache_bypasses: Dict[str, int] = defaultdict(int)
    # Every time you retrace using the same fake tensor mode, you should
    # advance the epoch so we don't reuse unbacked memos
    epoch: int = 0
    in_kernel_invocation: bool = False

    def __init__(
        self,
        *,
        allow_fallback_kernels=True,
        allow_non_fake_inputs=False,
        shape_env=None,
        static_shapes=None,
        # TODO: This is a temporary measure, see
        # https://github.com/pytorch/pytorch/pull/126245#discussion_r1604185748
        # We're currently solely using this to impede population of
        # item_memo for 0d scalar tensor inputs when export, because this
        # causes things that used to be deferred runtime asserts to turn into
        # guards, and then the guards are just lost.  We can potentially fix
        # this by ensuring guards also get put in the graph, but this is
        # pending a rework of how deferred runtime asserts in export.  Once
        # that's done, we can remove this.
        export=False,
    ):
        log.debug("create_mode 0x%x", id(self))
        self.allow_fallback_kernels = allow_fallback_kernels

        import torch._dynamo.config
        import torch._functorch.config

        self.propagate_real_tensors = (
            torch._functorch.config.fake_tensor_propagate_real_tensors
        )
        self.fake_tensor_converter = FakeTensorConverter(
            copy_data=self.propagate_real_tensors,
            export=export,
        )

        if static_shapes is not None:
            self.static_shapes = static_shapes
        else:
            self.static_shapes = shape_env is None

        # This is temporarily patched to True in Dynamo to grandfather in some
        # places where we unconditionally allow scalar outputs, TO BE REMOVED
        self.allow_scalar_outputs = False

        self._allow_unsafe_data_ptr_access = (
            torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access
        )
        self.allow_meta = torch._functorch.config.fake_tensor_allow_meta
        self.cache_enabled = (
            torch._dynamo.config.fake_tensor_cache_enabled
            and not self.propagate_real_tensors
        )
        self.cache_crosscheck_enabled = (
            torch._dynamo.config.fake_tensor_cache_crosscheck_enabled
        )

        # A flag that controls, whether we want to invoke ops on mix of
        # real weights/global variables and fake inputs
        self.allow_non_fake_inputs = allow_non_fake_inputs

        # [in_kernel_invocation]
        # when FakeTensor is invoked in user code, .device should return
        # the fake_device of the tensor so that code such as as `if x.is_cuda`
        # or torch.zeros([10, 10], device=x.device) continues to execute as if
        # the FakeTensor were real. However, within kernel execution, we return
        # the `Meta` device because all computation within the kernels should
        # behave as if the Tensors are on meta devices. Kernels should allocate
        # new tensors on meta devices, and checks like `is_meta` should return true.
        # within python refs, we always return the real device by defining
        # the device property
        self.in_kernel_invocation = False

        # True if we enter'ed and actually enabled fake tensor mode,
        # false if it was a no-op.  Not thread safe but neither is
        # in_kernel_invocation
        # If another fake mode was already active when we enter, we also stash it here.
        # That way when we exit, we know to re-enable the previous fake mode.
        self.enter_stack: List[
            Tuple[bool, Optional[TorchDispatchMode], Optional[_bool]]
        ] = []

        self.shape_env: ShapeEnv = shape_env

        self._stack_trace = traceback.extract_stack()
        self._stack = None

        # Indicates to our torch_dispatch dispatching infra that
        # this is an "infra" mode with lower dispatching precedence.
        self._mode_key = torch._C._TorchDispatchModeKey.FAKE

    # Typically, there is only one fake tensor mode and you test for it by
    # doing an isinstance test.  However, in some situations, there might be
    # TWO fake tensor modes.  The canonical example of this is exporting
    # a fake model: there is an outer fake mode created by the user, and
    # an inner fake mode created by Dynamo.  The two phase process is required
    # because the outer fake mode typically won't have a ShapeEnv, even if
    # the user is interested in exporting with dynamic shapes (so the inner
    # fake mode will actually have a ShapeEnv and swap in symbolic sizes.)
    #
    # In this case, it's insufficient to test only one FakeTensor: you need
    # to distinguish between our fake tensor and other fake tensors.  That's
    # what this function does.
    def is_our_fake(self, t):
        return isinstance(t, FakeTensor) and t.fake_mode is self

    # If we should avoid device init. This changes the behavior of various APIs:
    # - We avoid constant-prop on Tensors with ops that move them to another device
    # - We change the torch.tensor ctor contract to never materialize
    #   tensors on device
    #   (see NOTE: [torch.tensor, lift_fresh, and device movement])
    @property
    def avoid_device_init(self):
        return not torch.cuda.is_available()

    @property
    def stack(self):
        if self._stack is None:
            self._stack = "".join(traceback.format_list(self._stack_trace))
        return self._stack

    @count
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # FakeTensorMode should not be set when we're inside of it.
        assert (
            torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is None
        ), func
        try:
            return self.dispatch(func, types, args, kwargs)
        except TypeError:
            log.exception("fake tensor raised TypeError")
            raise

    # No-op if FakeTensorMode is already in use
    def __enter__(self):
        prev_only_lift_cpu_tensors = None
        if self.avoid_device_init:
            # See NOTE: [torch.tensor, lift_fresh, and device movement]
            prev_only_lift_cpu_tensors = torch._C._only_lift_cpu_tensors()
            torch._C._set_only_lift_cpu_tensors(True)
        maybe_prev_fake_mode = torch._C._unset_dispatch_mode(self._mode_key)
        if self is not maybe_prev_fake_mode:
            self.enter_stack.append(
                (True, maybe_prev_fake_mode, prev_only_lift_cpu_tensors)
            )
            return super().__enter__()
        else:
            # no-op (still need to re-set the fake mode though since we unset it)
            torch._C._set_dispatch_mode(self)
            self.enter_stack.append((False, None, prev_only_lift_cpu_tensors))
        return self

    def __exit__(self, a, b, c):
        (
            live,
            maybe_prev_fake_mode,
            maybe_prev_only_lift_cpu_tensors,
        ) = self.enter_stack.pop()
        if live:
            out = super().__exit__(a, b, c)
            # Re-enable the previous fake mode, if there was one.
            if maybe_prev_fake_mode is not None:
                torch._C._set_dispatch_mode(maybe_prev_fake_mode)
            if maybe_prev_only_lift_cpu_tensors is not None:
                torch._C._set_only_lift_cpu_tensors(maybe_prev_only_lift_cpu_tensors)

    @classmethod
    def cache_info(cls) -> DispatchCacheInfo:
        """
        Query the state of the dispatch cache.
        """
        return DispatchCacheInfo(
            FakeTensorMode.cache_hits,
            FakeTensorMode.cache_misses,
            dict(FakeTensorMode.cache_bypasses),
            len(FakeTensorMode.cache),
        )

    @classmethod
    def cache_clear(cls):
        """
        Clear the dispatch cache.
        """
        cls.cache_hits = 0
        cls.cache_misses = 0
        cls.cache_bypasses.clear()
        cls.cache.clear()

    def _cached_dispatch_impl(
        self,
        func: OpOverload,
        types: Tuple[Any, ...],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Lookup a cache entry for the given arguments. If none exists, dispatch
        and cache the result (if the result is eligible for caching).
        """
        output: Union[FakeTensor, _Unassigned] = _UNASSIGNED
        try:
            key = self._cache_key(func, args, kwargs)
            entry = FakeTensorMode.cache.get(key, None)
            if entry is not None:
                output = self._output_from_cache_entry(entry, func, args)
                FakeTensorMode.cache_hits += 1
                if self.cache_crosscheck_enabled:
                    # For debugging / testing: Validate that the output synthesized
                    # from the cache matches the output created by normal dispatch.
                    self._crosscheck_cache_output(output, func, types, args, kwargs)
            else:
                self._validate_cache_key(func, args, kwargs)
                output = self._dispatch_impl(func, types, args, kwargs)
                entry = self._make_cache_entry(key, func, args, kwargs, output)
                FakeTensorMode.cache[key] = entry
                FakeTensorMode.cache_misses += 1
        except _BypassDispatchCache as e:
            FakeTensorMode.cache_bypasses[e.reason] += 1

        if output is _UNASSIGNED:
            output = self._dispatch_impl(func, types, args, kwargs)

        return output

    def _cache_key(
        self,
        func: OpOverload,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> _DispatchCacheKey:
        """
        Create a cache key given the dispatch args. Raises _BypassDispatchCache
        for any situation that precludes caching.
        """
        key_values = (
            func,
            # Translate any FakeTensor args to metadata.
            self._prep_args_for_hash(args) if args else (),
            self._prep_args_for_hash(kwargs) if kwargs else (),
            # Capture the default_dtype mode since that can affect the output tensor,
            # e.g., when operating on constant float values.
            torch.get_default_dtype(),
            # Capture the current device to support, e.g., cache tensor creation,
            # where there isn't necessarily a tensor to take the device from.
            torch._C._get_default_device(),
            # We want to create tensors from cached metadata only when the inference
            # mode is the same.
            torch.is_inference_mode_enabled(),
            # Shape env settings could affect behavior. One example seen in the wild:
            # Disallowing dynamic shapes can introduce a DynamicOutputShapeException
            # where it wasn't seen on a previous instance of the same op.
            self.shape_env.settings if self.shape_env else None,
        )
        return _DispatchCacheKey(key_values)

    def _validate_cache_key(
        self,
        func: OpOverload,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Validate that the cache key generated by _cache_key will be
        reasonable.
        """
        # Avoid caching for any ops that would require a more sophisticated
        # caching implementation, e.g., data dependent ops or ops that modify
        # the inputs.
        if torch.Tag.data_dependent_output in func.tags:
            raise _BypassDispatchCache("data dependent output")

        if torch.Tag.dynamic_output_shape in func.tags:
            raise _BypassDispatchCache("dynamic output shape")

        if torch.Tag.inplace_view in func.tags:
            raise _BypassDispatchCache("inplace view")

        if func == aten._unsafe_view.default:
            raise _BypassDispatchCache("unsafe view")

        if func in self.lift_fns:
            raise _BypassDispatchCache("lift")

        if func.name() == "inductor::resize_storage_bytes_":
            raise _BypassDispatchCache("inductor::resize_storage_bytes_")

        if not torch._library.utils.is_builtin(func):
            raise _BypassDispatchCache("non-builtin")

        # In order to handle storage aliasing, we need to establish the alias
        # for any view op on a cache hit. But CompositeImplicitAutograd ops may
        # or may not alias the input, so just punt on caching these.
        if func.is_view and torch._C._dispatch_has_kernel_for_dispatch_key(
            func.name(), torch._C.DispatchKey.CompositeImplicitAutograd
        ):
            raise _BypassDispatchCache("CompositeImplicitAutograd")

    def _prep_args_for_hash(self, args: Any) -> Any:
        """
        Translate the provided args into a form suitable for caching at FakeTensor
        dispatch, i.e., convert unhashable types like lists & dicts into tuples and
        convert FakeTensors into metadata. Raises _BypassDispatchCache to signal
        unsupported cases that should bypass caching.
        """
        if isinstance(args, dict):
            args = list(args.keys()) + list(args.values())

        result: List[Any] = []
        for arg in args:
            if isinstance(arg, FakeTensor):
                if not self.is_our_fake(arg):
                    raise _BypassDispatchCache("not our fake")
                if arg._has_symbolic_sizes_strides:
                    raise _BypassDispatchCache("symbolic shape")
                if arg.constant is not None:
                    raise _BypassDispatchCache("constant attribute")
                if arg.is_sparse:
                    raise _BypassDispatchCache("sparse tensor")
                if arg.layout in [
                    torch.sparse_csr,
                    torch.sparse_csc,
                    torch.sparse_bsr,
                    torch.sparse_bsc,
                ]:
                    # Does this subsume arg.is_sparse?
                    raise _BypassDispatchCache("sparse tensor layout")
                # sparse tensors don't have storage, so check is after
                if isinstance(arg.untyped_storage().nbytes(), torch.SymInt):
                    raise _BypassDispatchCache("symbolic nbytes")
                if is_sparse_compressed(arg):
                    raise _BypassDispatchCache("sparse compressed tensor")
                result.append(extract_tensor_metadata(arg))
            elif isinstance(arg, torch.Tensor):
                raise _BypassDispatchCache("non-fake tensor")
            elif isinstance(arg, (torch.SymBool, torch.SymInt, torch.SymFloat)):
                raise _BypassDispatchCache("symbolic shape")
            elif isinstance(arg, (list, tuple, dict)):
                result.extend(self._prep_args_for_hash(arg))
            else:
                # It's important to capture the type of the arg since, e.g., 1 and 1.0
                # hash to the same value, but can produce different dtypes for the
                # output tensor.
                result.append((type(arg), arg))

        return tuple(result)

    def _make_cache_entry(
        self,
        key: _DispatchCacheKey,
        func: OpOverload,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        output: FakeTensor,
    ) -> _DispatchCacheEntry:
        """
        Make a cache entry object for the given 'output' Tensor. Raises
        _BypassDispatchCache if the output tensor has characteristics that
        prevent caching it.
        """
        # Some ops return tuples of Tensors, but it's rare, so avoid
        # the complexity of caching other types.
        if not isinstance(output, FakeTensor):
            raise _BypassDispatchCache("non-FakeTensor output")

        # Avoid caching FakeTensors with constants attached since those
        # can be invalidated.
        if output.constant is not None:
            raise _BypassDispatchCache("constant attribute")

        # TODO: support caching sparse outputs?
        if output.is_sparse:
            raise _BypassDispatchCache("sparse output")

        if is_sparse_compressed(output):
            raise _BypassDispatchCache("sparse compressed output")

        # Can an in-place op really reference a kwarg? If so, then we need
        # to extend the implementation to handle it.
        for kval in kwargs.values():
            if id(kval) == id(output):
                raise _BypassDispatchCache("kwarg aliases output")

        # If this is an in-place op, the entry records which input arg is aliased.
        for idx in range(len(args)):
            if id(args[idx]) == id(output):
                return _DispatchCacheEntry(
                    inplace_idx=idx, metadata=None, view_idx=None
                )

        # Otherwise, create an entry that records the output tensor's metadata.
        view_idx = None
        if func.is_view:
            idxs = [i for i, t in enumerate(args) if isinstance(t, torch.Tensor)]
            assert len(idxs) == 1
            view_idx = idxs[0]

        metadata = extract_tensor_metadata(output)
        entry = _DispatchCacheEntry(
            inplace_idx=None, metadata=metadata, view_idx=view_idx
        )

        # N.B.: Some checks for bypassing the cache would be performed on the
        # output tensor synthesized from the cached metadata. As an optimization,
        # we can synthesize a tensor here and do the checks on that instance.
        # This approach keeps the (more frequent) cache-hit path as lightweight
        # as possible.
        synth_output = self._output_from_cache_entry(entry, func, args)

        # Make sure the dispatch_key_set from the synthesized output tensor will
        # be the same.
        synth_key_set = torch._C._dispatch_key_set(synth_output)
        key_set = torch._C._dispatch_key_set(output)
        if synth_key_set != key_set:
            raise _BypassDispatchCache("dispatch_key_set mismatch")

        return entry

    def _output_from_cache_entry(
        self, entry: _DispatchCacheEntry, func: OpOverload, args: Tuple[Any, ...]
    ) -> FakeTensor:
        """
        Create a new FakeTensor from the cache entry.
        """
        if entry.inplace_idx is not None:
            # This is an in-place op; return the aliased arg.
            return args[entry.inplace_idx]

        # Synthesize a new FakeTensor with the cached metadata.
        metadata = entry.metadata
        assert metadata and not metadata.is_sparse

        empty = torch.empty_strided(
            metadata.shape,
            metadata.stride,
            dtype=metadata.dtype,
            layout=metadata.layout,
            device="meta",
            requires_grad=metadata.requires_grad,
        )

        if metadata.is_conj:
            torch._C._set_conj(empty, True)
        if metadata.is_neg:
            torch._C._set_neg(empty, True)

        maybe_suppress: Callable[[], Any] = contextlib.nullcontext
        if self.shape_env is not None:
            maybe_suppress = self.shape_env.suppress_guards

        if func.is_view:
            # For view ops, the storage should be the same as the tensor input.
            storage = args[cast(int, entry.view_idx)].untyped_storage()
            with in_kernel_invocation_manager(self), maybe_suppress():
                empty.set_(
                    storage, metadata.storage_offset, metadata.shape, metadata.stride
                )
        elif metadata.storage_offset != 0:
            storage = empty.untyped_storage()
            with in_kernel_invocation_manager(self), maybe_suppress():
                empty.set_(
                    storage, metadata.storage_offset, metadata.shape, metadata.stride
                )
        if metadata.storage_bytes == 0:
            empty.untyped_storage().resize_(0)

        return FakeTensor(self, empty, metadata.device)

    def _crosscheck_cache_output(
        self,
        output: FakeTensor,
        func: OpOverload,
        types: Tuple[Any, ...],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Helper to validate that the output synthesized from the cache matches
        the output created by normal dispatch.
        """
        try:
            true_output = self._dispatch_impl(func, types, args, kwargs)
        except Exception as e:
            raise RuntimeError(
                f"FakeTensor cache crosscheck failure: func={func}, "
                f"args={args}, kwargs={kwargs}: Dispatch raised={e}"
            ) from e
        try:
            assert_metadata_eq(assert_eq, true_output, output)
        except Exception as e:
            raise RuntimeError(
                f"FakeTensor cache crosscheck failure: func={func}, "
                f"args={args}, kwargs={kwargs}"
            ) from e

    def dispatch(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        with no_dispatch():
            log.debug("%s %s %s", func, args, kwargs)

        if func in _DISPATCH_META_HANDLERS:
            return _DISPATCH_META_HANDLERS[func](args)

        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(
                "%sFakeTensorMode.__torch_dispatch__: %s", " " * RECURSION_COUNT, func
            )
            # NOTE: incr is intentionally unused for a RAII pattern
            incr = IncrementRecursionCount()

        # Some attribute queries that can be serviced directly
        # See Note [is_coalesced is dispatched]
        if func in _DISPATCH_HANDLE_DIRECTLY:
            # NB: no_dispatch is ok here too, this func is very simple
            with in_kernel_invocation_manager(self):
                return func(*args, **kwargs)

        if self.cache_enabled:
            return self._cached_dispatch_impl(func, types, args, kwargs)
        else:
            return self._dispatch_impl(func, types, args, kwargs)

    def _dispatch_impl(self, func, types, args, kwargs) -> FakeTensor:
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))

        flat_arg_fake_tensors = [t for t in flat_args if self.is_our_fake(t)]
        has_symbolic_sizes = any(
            i._has_symbolic_sizes_strides for i in flat_arg_fake_tensors
        ) or any(isinstance(a, torch.SymInt) for a in flat_args)

        converter = self.fake_tensor_converter

        is_lift_func = func in self.lift_fns

        # To constant propagate through these functions:
        # 1, If this is a lift due to a torch.tensor call,
        #    the input tensor is guaranteed to be a
        #    constant, so we keep a copy of the original argument along so
        #    we can query it if we're asked to item() it at some later point.
        #    (Note that you can always call a lift fn manually, so we do
        #    have to check if there are any fake tensors!)
        # 2, Some functions that allow Python numbers to bind to Tensors, e.g, torch.div
        if (is_lift_func and not flat_arg_fake_tensors) or (
            should_allow_numbers_as_tensors(func)
            and not has_symbolic_sizes
            and not flat_arg_fake_tensors
        ):
            assert all(
                t.constant is not None for t in flat_arg_fake_tensors
            ), f"{func} should not have fake inputs without constants"
            const_flat_args = [
                a.constant if self.is_our_fake(a) else a for a in flat_args
            ]
            const_args, const_kwargs = pytree.tree_unflatten(const_flat_args, args_spec)
            out = func(*const_args, **const_kwargs)
            if type(out) is torch.Tensor and self.may_turn_const(out):
                # NB: not in_kernel_invocation_manager because we're doing real
                # compute here
                # NB: no_dispatch() here is VERY DANGEROUS (like, segfault
                # dangerous) if this is actually a wrapper subclass tensor,
                # therefore the exact type test above
                with no_dispatch():
                    out = out.clone()
                return converter.from_real_tensor(self, out, make_constant=True)

        # See [subclass inputs] below
        # NB: If you're seeing a mysterious infinite loop involving fake
        # tensor, it might be related to this line.  Though I'm not sure
        # how you'll know to read this comment, as this line won't show up
        # in the stack trace.
        has_unrecognized_types = _check_for_subclass(flat_args)
        if has_unrecognized_types:
            unrecognized_types = [
                type(x) for x in flat_args if _check_for_subclass_arg(x)
            ]
            not_implemented_log.debug(
                "FakeTensorMode unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        # if we are in the dispatch mode, we will enter this function even if the inputs
        # are not FakeTensors. For now, throw if any non-Fake Tensor inputs
        # and just support constructors.

        # this is generated from torch.tensor(), which does not use the
        # dispatcher, to allow wrapper subclasses to wrap the new tensor
        if is_lift_func:
            assert len(kwargs) == 0 and len(args) == 1, f"{args} {kwargs}"

            if type(args[0]) is torch.Tensor:
                return converter.from_real_tensor(self, args[0])

        # If we are trying to avoid device init, then we need to avoid constant
        # prop on constant tensors for ops that change devices.
        avoiding_device_init = False
        if self.avoid_device_init:
            if (
                func == torch.ops.aten._to_copy.default
                and "device" in kwargs
                and kwargs["device"] != "cpu"
            ):
                avoiding_device_init = True
            if func == torch.ops.prims.device_put.default:
                avoiding_device_init = True

        # Recompute flat_arg_fake_tensors here again in case some of the inputs
        # were real tensors and fakified in validate_and_convert_non_fake_tensors
        (flat_args, flat_arg_fake_tensors) = self.validate_and_convert_non_fake_tensors(
            func, converter, flat_args, args_spec
        )
        del args, kwargs  # Invalidated

        # The current constant handling only support tracing systems
        # (aot autograd, torchdynamo) where each operation is run consecutively.
        # Because each operation is run in order, we can trace out and support
        # sequences like: x = torch.tensor(0.); y = x.add_(1)
        # Whenver a constant is written to but with inputs that cannot be evaluated
        # statically, such as random_(), we invalidate all constants that alias the input
        # We will rely on functionalization for use of fake tensors constants as persistent
        # objects on an FX Graph.

        # We dispatch size/stride/numel on the FakeTensor not its constant, so bail on inplace_view
        all_constant = all(e.constant is not None for e in flat_arg_fake_tensors)
        if (
            torch.Tag.nondeterministic_seeded not in func.tags
            and torch.Tag.inplace_view not in func.tags
            and all_constant
            and len(flat_arg_fake_tensors) != 0
            and not has_symbolic_sizes
            and not avoiding_device_init
        ):
            const_flat_args = [
                a.constant if self.is_our_fake(a) else a for a in flat_args
            ]
            const_args, const_kwargs = pytree.tree_unflatten(const_flat_args, args_spec)

            # NB: not in_kernel_invocation_manager(self) as we want to do REAL
            # compute
            with no_dispatch():
                out = func(*const_args, **const_kwargs)

            flat_out = pytree.tree_leaves(out)
            flat_out_tensors = [t for t in flat_out if isinstance(t, torch.Tensor)]
            all_constant = all(self.may_turn_const(t) for t in flat_out_tensors)

            if all_constant:
                return pytree.tree_map_only(
                    torch.Tensor,
                    lambda t: converter.from_real_tensor(self, t, make_constant=True),
                    out,
                )

            # we weren't able to turn outputs to constants,
            # so invalidate all constants that might be aliases of the outputs
            for ten in flat_out_tensors:
                converter.invalidate_constant_aliases(ten)

        # we are falling through to running non constant tensors, any input constant that
        # is written to must be invalidated
        args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
        self.invalidate_written_to_constants(func, flat_arg_fake_tensors, args, kwargs)

        def maybe_to_real_tensor(t):
            if isinstance(t, FakeTensor):
                return t.real_tensor
            elif isinstance(t, SymTypes):
                return t.node.pytype(
                    t.node.expr.xreplace(self.shape_env.var_to_val).xreplace(
                        self.shape_env.unbacked_var_to_val
                    )
                )
            else:
                return t

        from torch.fx.experimental.symbolic_shapes import (
            compute_unbacked_bindings,
            free_unbacked_symbols,
            SymTypes,
        )

        nil = object()

        real_out = nil
        if (
            self.propagate_real_tensors
            and all(e.real_tensor is not None for e in flat_arg_fake_tensors)
            # TODO: Handle SymFloat/SymBool
            and not any(
                (
                    isinstance(a, torch.SymInt)
                    and (syms := free_unbacked_symbols(a))
                    and any(s not in self.shape_env.unbacked_var_to_val for s in syms)
                )
                for a in flat_args
            )
        ):
            real_flat_args = [maybe_to_real_tensor(a) for a in flat_args]
            real_args, real_kwargs = pytree.tree_unflatten(real_flat_args, args_spec)
            real_out = func(*real_args, **real_kwargs)
        elif self.propagate_real_tensors:
            # This can happen occasionally legitimately, specifically when you
            # are inside the meta of a data dependent operation and you create
            # a tensor on an unbacked SymInt; at this point in time we don't
            # know what the unbacked SymInt is, but we will know later.
            # However, if there's a bug in the condition above, this condition
            # will also trigger.
            log.debug(
                "propagate_real_tensors skipped %s(%s, %s) %s",
                func,
                flat_arg_fake_tensors,
                flat_args,
                self.shape_env.unbacked_var_to_val if self.shape_env else None,
            )

        def maybe_propagate_real_tensors(fake_out):
            import sympy

            def go(t, real_t):
                if isinstance(t, FakeTensor):
                    # NB: unconditionally overwrite
                    t.real_tensor = real_t
                elif isinstance(t, SymTypes) and free_unbacked_symbols(t):
                    if isinstance(t.node.expr, sympy.Symbol):
                        self.shape_env.set_unbacked_var_to_val(t.node.expr, real_t)

            if real_out is not nil:
                tree_map_(go, fake_out, real_out)

                # If a data-dependent op is used in a decomposition, we
                # may need to get the unbacked settings "early"
                # TODO: Is this really needed?
                compute_unbacked_bindings(self.shape_env, fake_out, peek=True)

            return fake_out

        # Try for fastpath
        if has_symbolic_sizes:
            fast_impl = get_fast_op_impls().get(func)
            if fast_impl is not None:
                return maybe_propagate_real_tensors(fast_impl(self, *args, **kwargs))

        # If there's a Python meta, prefer that over the decomposition
        from torch._decomp import meta_table as meta_table

        if func not in meta_table and not self.cpp_meta_supports_symint(func):
            from torch._decomp import decomposition_table

            # Prefer Python decompositions over C++ ones
            if func in decomposition_table and (
                has_symbolic_sizes
                or (
                    # TODO: Remove these exclusions, so that we can remove
                    # this leg entirely
                    torch_decomp_decompositions(func)
                    and all(not e.is_sparse for e in flat_arg_fake_tensors)
                )
            ):
                with self:
                    return decomposition_table[func](*args, **kwargs)

            with self:
                # Decomposes CompositeImplicitAutograd ops
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        # prims already wrap FakeTensor inputs to FakeTensor outputs
        # and do device logic, we dont need do anything but run them
        # and ensure that Meta kernels are dispatched to (see)
        # Fake Tensor Dispatch Keys
        # TODO - we should be use the prim aten impl
        # TODO - fix prims complex ops
        if (
            "prims::" in func._schema.name
            and hasattr(func, "prim_meta_impl")
            and not stride_incorrect_op(func)
        ):
            with self:
                return maybe_propagate_real_tensors(
                    func.prim_meta_impl(*args, **kwargs)
                )

        # Users can register FakeTensor rules for custom operators
        # Call them if they exist.
        maybe_fake_impl = torch._library.simple_registry.singleton.find(
            func.name()
        ).fake_impl.kernel
        if maybe_fake_impl:
            ctx = torch._library.fake_impl.FakeImplCtx(self, func)
            with torch._library.fake_impl.set_ctx_getter(lambda: ctx), self:
                result = maybe_fake_impl(*args, **kwargs)
                return maybe_propagate_real_tensors(result)

        # special handling for funcs registered through `register_op_impl`,
        # e.g., manipulating args on constructor calls to construct meta tensors
        # and then afterwards wrapping them to a FakeTensor
        for run_impl_check, op_impl in op_implementations_checks:
            if run_impl_check(func):
                op_impl_out = op_impl(self, func, *args, **kwargs)
                if op_impl_out is not NotImplemented:
                    return maybe_propagate_real_tensors(op_impl_out)

        def maybe_run_unsafe_fallback(error=None):
            # We infer the meta of a custom ops that return None to just
            # return None. custom ops are not allowed to mutate metadata
            # of their inputs, so this is safe.
            if torch._library.utils.can_generate_trivial_fake_impl(func):
                return None
            # no meta kernel registered, fallback to kernel for the device
            if has_symbolic_sizes or not self.can_run_unsafe_fallback(func):
                raise UnsupportedOperatorException(func)
            if error is None:
                error = UnsupportedOperatorException(func)
            return run_fallback_kernel(self, func, flat_args, args_spec, error)

        # Optimization: If there is no Meta kernel, it takes a surprisingly long
        # amount of time to catch the NotImplementedError, so we check it here.
        if not has_meta(func):
            return maybe_propagate_real_tensors(maybe_run_unsafe_fallback())

        # run kernel registered to meta for func, which include
        # python meta registrations, prims, decomps, and c++ meta fns (structured kernels)
        # It's possible that the kernel will return NotImplementedError
        try:
            with in_kernel_invocation_manager(self):
                r = func(*args, **kwargs)
        except NotImplementedError as not_implemented_error:
            return maybe_run_unsafe_fallback(not_implemented_error)
        except Exception:
            log.exception("failed while attempting to run meta for %s", func)
            raise

        return maybe_propagate_real_tensors(
            self.wrap_meta_outputs_with_default_device_logic(
                r, func, flat_args, device=kwargs.get("device")
            )
        )

    # WARNING: DO NOT add any additional namespaces/operators here if they refer to operators
    # outside of the pytorch/pytorch library! Any pre-existing things here
    # are either in the pytorch/pytorch library or have been grandfathered in.
    # The fallback does not always work and MAY CRASH and emit unreadable error messages
    # so it should not be allowed by default.
    _can_run_unsafe_fallback_allowed_namespaces = ordered_set(
        "debugprims",
        "prims",
        "aten",
        "xla",
        "vision",
        "torchtext",
        "torchaudio",
        "quantized",
    )

    def can_run_unsafe_fallback(self, func: OpOverload):
        if not self.allow_fallback_kernels:
            return False
        # It's OK to try the fallback for built-in ops (e.g. aten, prims)
        # because we control and test these but the fallback leads to unexpected behavior
        # in user-defined custom ops
        return (
            func.namespace in self._can_run_unsafe_fallback_allowed_namespaces
            or func.name() == "fbgemm::gmm"
        )

    def validate_and_convert_non_fake_tensors(
        self, func, converter, flat_args, args_spec
    ):
        """
        Checks if the list of tensors are fake tensors.
        If not, try to convert them to fake tensors.
        Returns the original args, kwargs, and a flattened list of (args, kwargs) that are fake tensors.
        """
        flat_arg_fake_tensors: List[Any] = []

        def validate(x):
            if not isinstance(x, torch.Tensor):
                return x

            nonlocal flat_arg_fake_tensors
            if not self.is_our_fake(x):
                if torch.Tag.inplace_view in func.tags:
                    args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                    raise AssertionError(
                        f"Can't call metadata mutating ops on non-Fake Tensor inputs. Found in {render_call(func, args, kwargs)}"
                    )
                if not self.allow_non_fake_inputs:
                    if isinstance(x, FakeTensor) and x.fake_mode is not self:
                        raise AssertionError("Mixing fake modes NYI")
                    args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                    raise AssertionError(
                        f"Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode "
                        f"with 'allow_non_fake_inputs'. Found in {render_call(func, args, kwargs)}"
                    )

                x = converter.from_real_tensor(self, x)

            flat_arg_fake_tensors.append(x)
            return x

        validated_args = [validate(a) for a in flat_args]
        return validated_args, flat_arg_fake_tensors

    def wrap_meta_outputs_with_default_device_logic(self, r, func, flat_args, device):
        converter = self.fake_tensor_converter

        # Lazily initialized, in case there are no tensor returns
        common_device = None
        has_scalar_only_inputs = False

        def wrap(e):
            nonlocal common_device
            nonlocal has_scalar_only_inputs

            if not isinstance(e, torch.Tensor):
                return e

            if common_device is None:
                (
                    common_device,
                    has_scalar_only_inputs,
                ) = FakeTensor._find_common_device(func, flat_args)

            is_our_fake = self.is_our_fake(e)
            if is_our_fake:
                torch._check(
                    e.device == common_device,
                    lambda: f"FakeTensor is wrapped to wrong device, found {e.device}, expected {common_device}",
                )
                return e
            elif converter is not None:
                if has_scalar_only_inputs:
                    # Under FakeTensorMode, op accepts scalar only inputs, such as aten.add/sub/mul/div,
                    # returns a real scalar tensor on CPU. See TensorMeta() in _prims/__init__.py for details.
                    # We thus directly convert real tensor to fake tensor.
                    return converter.from_real_tensor(self, e)
                else:
                    return converter.from_meta_and_device(
                        self, e, device or common_device
                    )
            else:
                return e

        return tree_map(wrap, r)

    _cpp_meta_supports_symint = ordered_set(
        aten.empty.memory_format,
        aten.empty_strided.default,
        aten.as_strided_scatter.default,
        aten.as_strided.default,
        aten.as_strided_.default,
        aten.zeros.default,
        aten.detach.default,
        aten.view_as_real.default,
        aten.view_as_complex.default,
        aten.set_.source_Storage_storage_offset,
        aten._sparse_coo_tensor_with_dims_and_tensors.default,
    )

    def cpp_meta_supports_symint(self, func):
        if torch.Tag.view_copy in func.tags:
            return True
        return func in self._cpp_meta_supports_symint

    lift_fns = ordered_set(aten.lift_fresh.default, aten.lift_fresh_copy.default)

    def may_turn_const(self, t):
        return (
            t.numel() <= CONSTANT_NUMEL_LIMIT
            and not t.is_sparse
            and not self.is_our_fake(t)
            and not t.device.type == "meta"
        )

    def invalidate_written_to_constants(
        self, func, flat_arg_fake_tensors, args, kwargs
    ):
        any_constant = any(e.constant is not None for e in flat_arg_fake_tensors)
        schema_info = get_schema_info(func)
        if any_constant and schema_info.is_mutable():
            _, new_kwargs = normalize_function(
                func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
            )
            for k, v in new_kwargs.items():
                k = k if (k != "input" or schema_info.has_argument(k)) else "self"
                if (
                    self.is_our_fake(v)
                    and schema_info.is_mutable(k)
                    and v.constant is not None
                ):
                    self.fake_tensor_converter.invalidate_constant_aliases(v.constant)

    def from_tensor(
        self,
        tensor,
        *,
        static_shapes=None,
        source: Optional[Source] = None,
        symbolic_context=None,
        trace=True,
    ):
        shape_env: Optional[ShapeEnv] = self.shape_env
        if static_shapes is None:
            static_shapes = self.static_shapes
        if static_shapes:
            assert (
                symbolic_context is None
            ), "cannot set both static_shapes and symbolic_context"
            shape_env = None
        return self.fake_tensor_converter.from_real_tensor(
            self,
            tensor,
            shape_env=shape_env,
            source=source,
            symbolic_context=symbolic_context,
            trace=trace,
        )