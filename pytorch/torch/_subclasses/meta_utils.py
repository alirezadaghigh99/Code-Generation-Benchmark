class MetaConverter:
    def __init__(self, *, copy_data: bool = False):
        # Maps MetaStorageId to UntypedStorage
        self.storage_memo: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        # Maps MetaTensorId to torch.Tensor (typically a meta tensor or
        # FakeTensor)
        self.tensor_memo: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.hit = 0
        self.miss = 0
        self.del_hook = None
        self.arg_cnt = 0
        # Ensures real_storage/real_tensor are populated on the resulting
        # metaified storage/tensor.  The naming of this attribute is load
        # bearing: FakeTensor relies on real tensor being set to exactly this
        # value
        self.copy_data = copy_data
        self.describer = MetaTensorDescriber(copy_data=copy_data)

    def successful(self):
        return self.hit > 0 and self.miss == 0

    def get_tensor_memo(self, t: MetaTensorDesc):
        return self.tensor_memo.get(t.id, None)

    def set_tensor_memo(self, t: MetaTensorDesc, v):
        self.tensor_memo[t.id] = v

    def get_storage_memo(self, s: MetaStorageDesc):
        return self.storage_memo.get(s.id, None)

    def set_storage_memo(self, s: MetaStorageDesc, v):
        self.storage_memo[s.id] = v

    def meta_storage(self, s: MetaStorageDesc, callback):
        # If we are fakeifying a tensor that has a secretly-zero-sized storage,
        # Need to make sure to resize the meta storage too.
        if self.get_storage_memo(s) is None:
            r_s = callback(
                lambda: torch.empty(s.size, dtype=torch.uint8, device="meta"),
            ).untyped_storage()
            if self.copy_data:
                # NB: no_dispatch is needed because internally storage copy is
                # implemented as Tensor operations
                with torch.no_grad(), no_dispatch():
                    assert s.data is not None
                    r_s.real_storage = s.data.clone()
            self.set_storage_memo(s, r_s)
            return r_s
        else:
            return self.get_storage_memo(s)

    # This function assumes that it's possible to do the conversion
    # NB: name here is used in a conventional way by Dynamo; it corresponds
    # precisely to the Source.name() of the tensor we're fakeifying and
    # corresponds to a valid Python expression.  When we construct sub-names
    # as part of this process, we will maintain this invariant!  (Even though
    # other users of this may not need it this property to be upheld.)
    def meta_tensor(
        self,
        t: MetaTensorDesc,
        shape_env: Optional[ShapeEnv] = None,
        callback=lambda t: t(),
        source: Optional[Source] = None,
        symbolic_context: Optional[SymbolicContext] = None,
    ):
        if source is None:
            from torch._dynamo.source import ConstantSource

            # TODO: make a dedicated UnknownSource for this?
            source = ConstantSource(
                f"__meta_utils_unknown_tensor{len(self.tensor_memo)}"
            )

        # This indicates you set no_dispatch() before calling into this
        # function.  This is an error: we may be creating fake tensors and
        # will perform operations on them which need fake tensor mode to
        # be active.  You will segfault if you are in a no_dispatch() block.
        assert not torch._C._dispatch_tls_local_exclude_set().has(
            torch._C.DispatchKey.Python
        )
        arg_cnt = self.arg_cnt
        self.arg_cnt += 1

        # When we make as_strided calls, we end up generating a guard
        # that the new as_strided tensor is in bounds for the old storage
        # for the base (since as_strided calls can "bust" out of their
        # bounding box.)  This guard is unnecessary: if a user is able
        # to provide us a tensor with the view base setup this way, we
        # don't need to produce a guard, because the fact that they
        # were able to produce the view base means its in bounds.
        #
        # Now, ordinarily, this guard would be harmless.  However, the
        # generated guard refers to variables bound on the base variable.
        # At the moment, Dynamo doesn't actually guard on x._base, because
        # according to Voz this results in a lot of spurious invalidations,
        # and also if the user doesn't directly make use of _base, its
        # pointless anyway (because programs should be parametric over
        # whether or not the input tensor is a view or not--unless you're
        # mutating the input, but that's a whole 'nother ballgame).  So
        # for expediency, we suppress these guards so we don't have to
        # deal with this (yet, anyway.)
        #
        # NB: An old version of this code suppressed guards for ALL operations
        # happening during meta conversion, not just as_strided calls.
        # This is too aggressive: we do duck sizing and 0/1 simplification
        # as we allocate variables, and we do need to register guards for
        # these cases.
        maybe_suppress: Callable[[], Any] = contextlib.nullcontext
        if shape_env is not None:
            maybe_suppress = shape_env.suppress_guards

        def sym_sizes_strides_storage_offset(
            t: MetaTensorDesc, src, symbolic_context=symbolic_context
        ) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
            assert t.stride is not None
            if shape_env is not None:
                fake_mode = t.fake_mode
                if fake_mode is not None and fake_mode.shape_env is shape_env:
                    # Don't reallocate the sizes; the shape envs are the same,
                    # so reuse the old sizes/strides/etc
                    return (t.size, t.stride, t.storage_offset)
                else:
                    # TODO: deduplicate this
                    t_size = tuple(
                        shape_env._maybe_specialize_sym_int_with_hint(sz)
                        for sz in t.size
                    )
                    t_stride = tuple(
                        shape_env._maybe_specialize_sym_int_with_hint(sd)
                        for sd in t.stride
                    )
                    t_storage_offset = shape_env._maybe_specialize_sym_int_with_hint(
                        t.storage_offset
                    )
                    return shape_env._create_symbolic_sizes_strides_storage_offset(
                        t_size,
                        t_stride,
                        t_storage_offset,
                        [d in t.dynamo_dynamic_indices for d in range(t.ndim)],
                        src,
                        symbolic_context=symbolic_context,
                    )
            else:
                return (t.size, t.stride, t.storage_offset)

        def empty_create(
            inner_t: MetaTensorDesc, inner_src, symbolic_context=symbolic_context
        ):
            (
                inner_sizes,
                inner_strides,
                inner_storage_offset,
            ) = sym_sizes_strides_storage_offset(inner_t, inner_src, symbolic_context)
            return torch.empty_strided(
                inner_sizes,
                inner_strides,
                dtype=inner_t.dtype,
                device="meta",
            )

        # Creates a subclass instance with empty inner tensors according to the specified
        # symbolic context.
        def empty_create_subclass(
            t: MetaTensorDesc,
            outer_size,
            outer_stride,
            symbolic_context=symbolic_context,
            callback=callback,
            source=source,
        ):
            from torch._dynamo.source import AttrSource
            from torch.fx.experimental.symbolic_shapes import SubclassSymbolicContext

            assert t.attrs is not None
            assert t.type is not None
            # NB: t.ctx could be None if the subclass in question has no
            # meaningful context

            # Note: transform_subclass will use __tensor_unflatten__ to generate
            # a fresh subclass wrapper with outer sizes / strides according to the
            # outer symbolic context (passed in to this function). Inner size / stride
            # / storage offset symbols are allocated according to the appropriate inner
            # symbolic contexts, after which the checks in transform_subclass() will
            # relate them to the outer metadata as possible.
            #
            # Morally, the code here is same as transform_subclass, but we've
            # written it from scratch to read EmptyCreateSubclass
            outer_size = outer_size if outer_size is not None else t.size
            outer_stride = outer_stride if outer_stride is not None else t.stride

            assert symbolic_context is None or isinstance(
                symbolic_context, SubclassSymbolicContext
            )

            def _empty_create_subclass(
                t, outer_size, outer_stride, symbolic_context, callback, source
            ):
                # We are hitting plain meta_desc tensor so actually
                # create a tensor here.
                if t.attrs is None:
                    r = callback(
                        lambda: empty_create(
                            t,
                            source,
                            symbolic_context,
                        )
                    )
                    if self.copy_data:
                        with torch.no_grad(), no_dispatch():
                            r.real_tensor = torch.empty_strided(
                                t.size,
                                t.stride,
                                dtype=t.dtype,
                                device=t.device,
                            )
                            assert t.data is not None
                            _safe_copy(r.real_tensor, t.data)
                    return r

                inner_tensors = {}
                for attr, meta_tensor_desc in t.attrs.items():
                    current_context = None
                    if symbolic_context is not None:
                        current_context = symbolic_context.inner_contexts[attr]

                    current_source = AttrSource(source, attr)
                    new_empty_tensor = _empty_create_subclass(
                        meta_tensor_desc,
                        meta_tensor_desc.size,
                        meta_tensor_desc.stride,
                        current_context,
                        callback,
                        current_source,
                    )
                    inner_tensors[attr] = new_empty_tensor

                return t.type.__tensor_unflatten__(
                    inner_tensors, t.ctx, outer_size, outer_stride
                )

            sub = _empty_create_subclass(
                t, outer_size, outer_stride, symbolic_context, callback, source
            )

            # NB: Purposefully guard here to simplify the inner / outer symbols.
            # Using sym_eq() for symbolic comparison can result in an expression that's too
            # difficult to guard on, so we use == here.
            assert sub.shape == outer_size, (
                f"Expected return value from {t.type}__tensor_unflatten__() to have "
                f"shape equal to {outer_size}, but got: {sub.shape}"
            )
            assert sub.stride() == outer_stride, (
                f"Expected return value from {t.type}__tensor_unflatten__() to have "
                f"stride equal to {outer_stride}, but got: {sub.stride()}"
            )

            return sub

        # Returns an all-dynamic symbolic context used for metafying the given tensor with
        # fully dynamic dims. This is useful when fake-ifying intermediate tensors in
        # closed-over ViewFunc state, as we don't have symbolic contexts for them, but we
        # don't want to over-specialize during view replay.
        def all_dynamic_symbolic_context(
            t: MetaTensorDesc, source, shape_env, callback
        ):
            from torch._dynamo.source import AttrSource
            from torch.fx.experimental.symbolic_shapes import (
                DimDynamic,
                StatelessSymbolicContext,
                SubclassSymbolicContext,
            )

            view_base_context: Optional[SymbolicContext] = None
            if t.is_view:
                assert t.base is not None
                view_base_context = all_dynamic_symbolic_context(
                    t.base, AttrSource(source, "_base"), shape_env, callback
                )

            t_symbolic_context: SymbolicContext
            t_dynamic_sizes = [DimDynamic.DYNAMIC] * t.ndim
            if t.is_traceable_wrapper_subclass:
                assert t.attrs is not None
                inner_contexts: Dict[str, SymbolicContext] = {}
                for attr, inner in t.attrs.items():
                    assert isinstance(attr, str)
                    inner_contexts[attr] = all_dynamic_symbolic_context(
                        inner, AttrSource(source, attr), shape_env, callback
                    )
                t_symbolic_context = SubclassSymbolicContext(
                    dynamic_sizes=t_dynamic_sizes,
                    constraint_sizes=[None] * t.ndim,
                    inner_contexts=inner_contexts,  # type: ignore[arg-type]
                    tensor_source=source,
                    view_base_context=view_base_context,
                )
            else:
                t_symbolic_context = StatelessSymbolicContext(
                    dynamic_sizes=t_dynamic_sizes,
                    constraint_sizes=[None] * t.ndim,
                    view_base_context=view_base_context,
                )

            return t_symbolic_context

        # Returns a fake-ified version of an input view tensor t, given an already fake-ified
        # base. At a high level, we want two things:
        #   1. fake_t should have the same view relationship to the given fake base as the
        #      input t has to its _base.
        #   2. fake_t should have symbolic sizes / strides / storage offset according to the
        #      appropriate symbolic context (i.e. from the automatic dynamic algorithm).
        #
        # We currently take different strategies across view types:
        #   * For dense -> dense views, accomplish both (1) and (2) simultaneously via an
        #     as_strided() call on the fake-ified base, passing symbolic metadata.
        #   * For views involving subclasses, perform view replay using view funcs to
        #     achieve (1). It's necessary for (2) to swap out any closed-over state in
        #     the view funcs with symbolicized SymInts and fake-ified tensors. Doing this
        #     avoids specialization (and thus over-eager simplification of symbols) that
        #     could occur during view replay on the fake-ified base.
        #
        # Examples:
        #   * t.unsqueeze(-1) with dense t is a dense -> dense view. It can be modeled
        #     with an as_strided() call on the fake base passing symbolic metadata.
        #   * sub.select(dim=0, index=3) is a subclass -> subclass view. The index arg
        #     is made symbolic to avoid invalid specialization and view replay is then
        #     done to reconstruct the view.
        #   * _nested_from_jagged(values, offsets) is a dense -> subclass view
        #     that returns a subclass instance from a dense values tensor. The offsets
        #     tensor is closed over in the view func, as it can be considered view metadata.
        #     First, the offsets tensor is fake-ified according to the inner symbolic
        #     context and with the correct relationship to the outer size / stride metadata.
        #     Then view replay is done, swapping in the fake offsets so the view replay output
        #     is fully fake with no invalid specialization.
        def view_from_base(
            base: torch.Tensor, t: MetaTensorDesc, source=source, shape_env=shape_env
        ):
            # fake-ify t's metadata according to the outer symbolic context
            (sizes, strides, storage_offset) = sym_sizes_strides_storage_offset(
                t, source
            )
            if (
                not t.is_traceable_wrapper_subclass
                and not is_traceable_wrapper_subclass(base)
            ):
                # Dense -> Dense view case uses as_strided() to construct view relationship.
                # TODO: Change this logic to use view replay for consistency?
                # It's likely there is no view func available.
                with maybe_suppress():
                    return base.as_strided(sizes, strides, storage_offset)

            from torch._dynamo.source import EphemeralSource
            from torch.fx.experimental.symbolic_shapes import (
                StatelessSymbolicContext,
                sym_eq,
            )

            def symint_visitor_fn(s):
                nonlocal symbolic_context
                from torch.fx.experimental.symbolic_shapes import DimDynamic

                all_static_sizes = (
                    symbolic_context is not None
                    and isinstance(symbolic_context, StatelessSymbolicContext)
                    and all(
                        x is DimDynamic.STATIC for x in symbolic_context.dynamic_sizes
                    )
                )
                # Can't just rely on shape env being None - dynamo always initializes it
                if all_static_sizes or shape_env is None:
                    return s

                # NB: The symbol here is expected to be simplified out because we a priori
                # allocate inner and outer symbols according to the appropriate symbolic
                # contexts and prefer those over this symbol during symbol simplification
                # (via usage of EphemeralSource below). This -shouldn't- happen, but if
                # this symbol somehow leaks out beyond the view tensor's shape metadata, our
                # assumption of it being simplified out will fail and it may be guarded on,
                # which will hard error.
                sym_source = EphemeralSource("symint_visitor_fn")

                symbol = shape_env.create_symbol(s, sym_source, positive=None)
                return shape_env.create_symintnode(symbol, hint=s, source=sym_source)

            real_to_fake_mapping = {}
            if t.is_traceable_wrapper_subclass:
                assert t.attrs is not None
                # NB: t.ctx could be None if the subclass in question has no
                # meaningful context
                assert t.type is not None

                # Fake-ify t naively here; this is only done so we can get fake-ified inner
                # tensors with the correct relationships to the outer sizes / strides for use
                # in view replay. It's done beforehand here because it's not easy to do when
                # visiting tensors one-by-one during view replay.
                #
                # Example:
                #   Consider a Dense -> NJT view. NJT has (values, offsets) components and we
                #   want a view of values with the offsets closed over. As the offsets component
                #   is needed to describe the output view, it's important that it's fakeified
                #   correctly.
                fake_t = empty_create_subclass(
                    t, outer_size=sizes, outer_stride=strides
                )
                attrs, _ = fake_t.__tensor_flatten__()
                for attr in attrs:
                    real_to_fake_mapping[t.attrs[attr].id] = getattr(fake_t, attr)

            def tensor_visitor_fn(
                visited_t: torch.Tensor,
                # These arguments are never passed, we just use them to close
                # over these relevant values
                shape_env=shape_env,
                callback=callback,
            ):
                # It's possible to close over an undefined tensor (e.g. NJT's lengths).
                if visited_t is None:
                    return None

                # NB: visited_t being a Tensor here is very naughty!  Should
                # have already been described

                # Fake inner tensors of view subclasses will come from the mapping built above.
                visited_id = self.describer.get_tensor_id(visited_t)
                fake_visited_t = real_to_fake_mapping.get(visited_id, None)
                if fake_visited_t is not None:
                    return fake_visited_t

                visited_desc = self.describer.describe_tensor(visited_t)

                # For other closed-over tensor state, fake-ify it as all dynamic with an
                # ephemeral source. This avoids invalid specialization during view replay.
                # If we find that in practice the usage of ephemeral sources isn't enough
                # to guarantee that we don't have guards on these symbols, we may need to
                # explicitly suppress guards (as is done for _base in the dense -> dense
                # view case).
                temp_source = EphemeralSource("tensor_visitor_fn")
                return self.meta_tensor(
                    visited_desc,
                    shape_env,
                    callback,
                    source=temp_source,
                    symbolic_context=all_dynamic_symbolic_context(
                        visited_desc, temp_source, shape_env, callback
                    ),
                )

            # Replay the view, swapping out any non-symbolic SymInts or real tensors
            # for symbolic SymInts or fake tensors.
            assert t.view_func is not None
            # NB: we do NOT suppress guards here, we need to remove ephemeral
            # sources
            fake_t = t.view_func(base, symint_visitor_fn, tensor_visitor_fn)

            # Ensure the output has symbolic shapes according to the outer symbolic context.
            # These checks should simplify out any symbols created for closed-over view func
            # SymInts.
            torch._check(sym_eq(fake_t.size(), sizes))
            torch._check(sym_eq(fake_t.stride(), strides))
            torch._check(sym_eq(fake_t.storage_offset(), storage_offset))
            return fake_t

        if self.get_tensor_memo(t) is None:
            GRAD_TENSOR_SENTINEL_VALUE = -2

            with torch.inference_mode(t.is_inference):
                if t.is_sparse:
                    is_leaf = t.is_leaf

                    # The lambda function below is similar to
                    # `t.to(device='meta')` except the latter
                    # preserves nnz value
                    r = callback(
                        lambda: torch.ops.aten._sparse_coo_tensor_with_dims(
                            t.sparse_dim,
                            t.dense_dim,
                            t.size,
                            dtype=t.dtype,
                            layout=torch.sparse_coo,
                            device="meta",
                        )
                    )
                    if self.copy_data:
                        # Pray that sparse clone doesn't lose information
                        assert t.data is not None
                        with torch.no_grad(), no_dispatch():
                            r.real_tensor = _safe_clone(t.data)
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    # Note [is_coalesced is dispatched]
                    # Strangely enough, is_coalesced() is a dispatched operator,
                    # which means that it will get caught by fake tensor mode.
                    # Ordinarily this would error, but there's some logic in
                    # fake tensor ensure this doesn't happen.
                    r._coalesced_(t.is_coalesced)
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        # This should probably use DelayedError,
                        # but clone is fine for now for sparse tensors.
                        # (DelayedError does not work for sparse because it causes
                        # the Fake sparse tensor to "lose" its fakeness)
                        r = r.clone()
                        with torch.enable_grad():
                            r._coalesced_(t.is_coalesced)
                elif is_sparse_compressed_layout(t.layout):
                    is_leaf = t.is_leaf

                    if t.layout in {torch.sparse_bsr, torch.sparse_bsc}:
                        assert t.sparse_dim is not None
                        assert t.dense_dim is not None
                        assert t.values is not None
                        batch_dim = t.ndim - t.sparse_dim - t.dense_dim
                        blocksize = t.values.shape[batch_dim + 1 : batch_dim + 3]
                    else:
                        blocksize = ()
                    if t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                        assert t.crow_indices is not None
                        index_dtype = t.crow_indices.dtype
                    else:
                        assert t.ccol_indices is not None
                        index_dtype = t.ccol_indices.dtype

                    r = callback(
                        lambda: torch.ops.aten._sparse_compressed_tensor_with_dims(
                            0,
                            t.dense_dim,
                            t.shape,
                            blocksize,
                            index_dtype,
                            layout=t.layout,
                            dtype=t.dtype,
                            device="meta",
                        )
                    )
                    if self.copy_data:
                        # Pray sparse clone doesn't lose information
                        assert t.data is not None
                        with torch.no_grad(), no_dispatch():
                            r.real_tensor = _safe_clone(t.data)
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        r = torch._C._functions.DelayedError(
                            "Internal error: Tried to backward() through example input",
                            1,
                        )(r)
                elif t.is_nested and not t.is_traceable_wrapper_subclass:
                    # TODO: Handle this better in Dynamo?
                    # There are checks there now, but this can still be triggered by a dense
                    # tensor graph input that is a view of a strided NT.
                    from torch._dynamo.exc import unimplemented

                    unimplemented(
                        "strided nested tensors are not supported by meta conversion"
                    )
                elif t.is_mkldnn:
                    is_leaf = t.is_leaf
                    sizes, strides, _storage_offset = sym_sizes_strides_storage_offset(
                        t, source
                    )
                    # TODO: This doesn't seem right, where's the MKLDNN'ness
                    # lol
                    r = callback(
                        lambda: torch.empty_strided(
                            sizes, strides, dtype=t.dtype, device="meta"
                        )
                    )
                    if self.copy_data:
                        with torch.no_grad(), no_dispatch():
                            assert t.size is not None
                            assert t.stride is not None
                            r.real_tensor = torch.empty_strided(
                                t.size, t.stride, dtype=t.dtype, device=t.device
                            )
                            assert t.data is not None
                            _safe_copy(r.real_tensor, t.data)
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        r = torch._C._functions.DelayedError(
                            "Internal error: Tried to backward() through example input",
                            1,
                        )(r)
                elif t.is_functorch_wrapped:
                    if t.is_view:
                        from torch._dynamo.exc import unimplemented

                        unimplemented(
                            "view functorch tensors are not supported by meta conversion"
                        )

                    # Wraps a functorch tensor class (BatchedTensor, GradTrackingTensor)
                    # in a FakeTensor
                    def _to_fake_tensor(t: MetaTensorDesc):
                        # TODO: why aren't the recursive calls going to
                        # meta_tensor
                        if t.is_batchedtensor:
                            assert t.unwrapped is not None
                            assert t.level is not None
                            assert t.bdim is not None
                            ft = _to_fake_tensor(t.unwrapped)
                            lvl = t.level
                            bdim = t.bdim
                            # You cannot create functorch tensors without
                            # having the ambient funtorch interpreter stack
                            # available, as the level refers to things in the
                            # stack
                            with torch._functorch.pyfunctorch.temporarily_restore_interpreter_stack(
                                t.functorch_stack
                            ):
                                r = _add_batch_dim(ft, bdim, lvl)
                        elif t.is_gradtrackingtensor:
                            assert t.unwrapped is not None
                            assert t.level is not None
                            disable_functorch = torch._C._DisableFuncTorch
                            with disable_functorch():
                                ft = _to_fake_tensor(t.unwrapped)
                            lvl = t.level
                            if lvl == GRAD_TENSOR_SENTINEL_VALUE:
                                r = ft
                            else:
                                with torch._functorch.pyfunctorch.temporarily_restore_interpreter_stack(
                                    t.functorch_stack
                                ):
                                    r = torch._C._functorch._wrap_for_grad(ft, lvl)

                            is_leaf = t.is_leaf
                            if t.requires_grad and safe_is_leaf(r):
                                r.requires_grad = True
                            elif t.requires_grad and not is_leaf:
                                r = torch._C._functions.DelayedError(  # type: ignore[assignment]
                                    "Internal error: Tried to backward() through example input",
                                    1,
                                )(
                                    r  # type: ignore[arg-type]
                                )
                        elif t.is_functional:
                            assert t.unwrapped is not None
                            assert t.current_level is not None
                            ft = self.meta_tensor(
                                t.unwrapped,
                                shape_env=shape_env,
                                callback=callback,
                                # NB: reuse these exactly, we treat the
                                # functional tensor as "invisible".
                                # TODO: Actually this all probably doesn't
                                # work, take a closer look.
                                source=source,
                                symbolic_context=symbolic_context,
                            )
                            r = _wrap_functional_tensor(ft, t.current_level)
                            # TODO: is_leaf/requires_grad?
                        else:
                            assert t.stride is not None

                            sizes = t.size
                            strides = t.stride
                            r = callback(
                                lambda: torch.empty_strided(
                                    sizes,
                                    strides,
                                    dtype=t.dtype,
                                    device="meta",
                                )
                            )
                            if self.copy_data:
                                with torch.no_grad(), no_dispatch():
                                    r.real_tensor = torch.empty_strided(  # type: ignore[attr-defined]
                                        t.size,
                                        t.stride,
                                        dtype=t.dtype,
                                        device=t.device,
                                    )
                                    assert t.data is not None
                                    _safe_copy(r.real_tensor, t.data)  # type: ignore[attr-defined]
                        return r

                    r = _to_fake_tensor(t)

                elif t.is_functional and t.device.type not in ["xla", "lazy"]:
                    assert t.unwrapped is not None
                    assert not t.is_functorch_wrapped  # handled above
                    unwrapped = self.meta_tensor(
                        t.unwrapped,
                        shape_env=shape_env,
                        callback=callback,
                        source=source,
                        symbolic_context=symbolic_context,
                    )
                    r = torch._to_functional_tensor(unwrapped)
                    torch._mirror_autograd_meta_to(t.autograd_meta_from, r)  # type: ignore[attr-defined]

                elif t.is_view:
                    # Construct views in two steps: recursively meta-fy their
                    # base, and then create view(s) off that.  NB: doing it
                    # directly from storage is WRONG because this won't cause
                    # version counters to get shared.

                    assert t.base is not None

                    base_symbolic_context = None
                    if shape_env and symbolic_context is not None:
                        from torch.fx.experimental.symbolic_shapes import (
                            StatelessSymbolicContext,
                        )

                        assert isinstance(symbolic_context, StatelessSymbolicContext)
                        # NB: This should generally be set when the input is a view,
                        # but the exception right now is for fake-ifying grads, which is
                        # a work in progress.
                        if symbolic_context.view_base_context is not None:
                            base_symbolic_context = symbolic_context.view_base_context

                    base = self.meta_tensor(
                        t.base,
                        shape_env,
                        callback,
                        source=torch._dynamo.source.AttrSource(source, "_base"),
                        symbolic_context=base_symbolic_context,
                    )

                    def is_c_of_r(complex_dtype, real_dtype):
                        return (
                            utils.is_complex_dtype(complex_dtype)
                            and utils.corresponding_real_dtype(complex_dtype)
                            == real_dtype
                        )

                    # In some situations, MetaConverter may be called in a
                    # context where autograd is disabled.  For the _is_view
                    # assert to pass, we have to setup the autograd view
                    # metadata anyway.  Do this by reenabling the
                    # ADInplaceOrView key.  This is kind of a hack.
                    old_exclude = torch._C._dispatch_tls_is_dispatch_key_excluded(
                        torch._C.DispatchKey.ADInplaceOrView
                    )
                    torch._C._dispatch_tls_set_dispatch_key_excluded(
                        torch._C.DispatchKey.ADInplaceOrView, False
                    )
                    try:
                        if base.dtype == t.dtype:
                            pass
                        elif is_c_of_r(base.dtype, t.dtype):
                            base = torch.view_as_real(base)
                        elif is_c_of_r(t.dtype, base.dtype):
                            base = torch.view_as_complex(base)
                        else:
                            # This is not guaranteed to succeed.  If it fails, it
                            # means there is another dtype-converting view function
                            # that hasn't been handled here
                            base = base.view(t.dtype)

                        # This is very tricky.  Naively, you might expect this
                        # to hold:
                        #
                        #   if t.requires_grad and not safe_is_leaf(t)
                        #       assert t._base.requires_grad
                        #
                        # But it's not true!  As you can see in the following
                        # program:
                        #
                        #   x = torch.zeros(4)
                        #   y = x.view(1, 4)
                        #   y.requires_grad = True
                        #   z = y.view(1, 1, 4)
                        #   assert z._base is x
                        #
                        # So we may have to do *two* views out of the base to
                        # recreate this situation.
                        if t.is_leaf:
                            # Leaf views that track view metadata are created by
                            # creating a view inside a no_grad block
                            with torch.no_grad():
                                r = view_from_base(base, t)
                            # As it's a leaf, we can directly assign requires_grad
                            r.requires_grad = t.requires_grad
                        else:
                            if t.base.requires_grad == t.requires_grad:
                                # Easy case, just run the view op
                                with torch.enable_grad():
                                    r = view_from_base(base, t)

                                # NB: We don't actaully faithfully replicate
                                # autograd connectivity, but that doesn't matter
                                # today. See following for more info:
                                # https://gist.github.com/soulitzer/e03f015b314c3f5fcf80888c69390913
                            else:
                                # Obscure case.  Create a leaf view and give it the
                                # correct requires_grad, then do the final view.
                                # NB: Can't have a non-leaf without requiring grad!
                                assert t.requires_grad
                                with torch.no_grad():
                                    mid = base.view(base.shape)
                                mid.requires_grad = t.requires_grad
                                with torch.enable_grad():
                                    r = view_from_base(mid, t)
                        # The CreationMeta influences whether or not inplace
                        # mutation is an error or not.  So we need to make
                        # sure we properly propagate this as well.
                        assert t.creation_meta is not None
                        torch._C._autograd._set_creation_meta(r, t.creation_meta)
                    finally:
                        torch._C._dispatch_tls_set_dispatch_key_excluded(
                            torch._C.DispatchKey.ADInplaceOrView, old_exclude
                        )

                else:
                    is_leaf = t.is_leaf

                    # Graph-Break for wrapped tensors
                    if (
                        not (t.is_batchedtensor or t.is_gradtrackingtensor)
                        and t.is_functorch_wrapped
                    ) or t.is_legacy_batchedtensor:
                        return NotImplemented

                    (
                        sizes,
                        strides,
                        storage_offset,
                    ) = sym_sizes_strides_storage_offset(t, source, symbolic_context)

                    # If we have a subclass that desugars into dense tensors,
                    # perform our callback on each inner tensor.
                    if t.is_traceable_wrapper_subclass:
                        r = empty_create_subclass(
                            t, outer_size=sizes, outer_stride=strides
                        )
                    else:
                        r = callback(
                            lambda: torch.empty_strided(
                                sizes,
                                strides,
                                dtype=t.dtype,
                                device="meta",
                            )
                        )
                        if self.copy_data:
                            with torch.no_grad(), no_dispatch():
                                assert t.size is not None
                                assert t.stride is not None
                                r.real_tensor = torch.empty_strided(
                                    t.size, t.stride, dtype=t.dtype, device=t.device
                                )
                                _safe_copy(r.real_tensor, t.data)

                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = t.requires_grad
                        if not is_leaf:
                            # Fake up some autograd history.
                            # Note: we *used* to call .clone() here to mock up some autograd history.
                            # This is bad for subclasses.
                            # Consider the case where you have a wrapper subclass that is contiguous,
                            # but its inner tensor is noncontiguous().
                            # .clone() (or other ops) will have the side effect of changing
                            # the metadata of the inner tensor.
                            # So instead, we now have a dedicated fn to set autograd history,
                            # without inadvertently changing other metadata.
                            r = torch._C._functions.DelayedError(
                                "Internal error: Tried to backward() through example input",
                                1,
                            )(r)

                    s = t.storage
                    assert s is not None
                    if s.id not in self.storage_memo and (
                        r.is_nested
                        or (
                            r.stride() == strides
                            and r.storage_offset() == storage_offset
                        )
                    ):
                        # You're normal and happy, install the fresh storage into the memo
                        self.set_storage_memo(s, r.untyped_storage())
                        if self.copy_data:
                            r.untyped_storage().real_storage = (
                                r.real_tensor.untyped_storage()
                            )
                    else:
                        # You're in crazy town; somehow you gave us a tensor
                        # that wasn't a view, but had nonzero storage offset,
                        # nontrivial strides (such that clone() couldn't
                        # preserve them), or already aliases with another
                        # tensor's storage.  The most typical way to end
                        # up here is with set_.  So use set_ to bludgeon this
                        # in.
                        r_s = self.meta_storage(s, callback=callback)
                        # NB: In principle, this should always work, but there
                        # is some subtle difference in the autograd metadata
                        # that means we will backprop the set_ call, even if
                        # r is declared as an input to grad.
                        # See https://github.com/pytorch/pytorch/issues/87956
                        # for the reproducer.
                        # NB: The in_kernel_invocation_manager here is necessary
                        # for fake tensor.  If we run the set_ call with fake
                        # tensor on, r will improperly report that it is NOT a
                        # meta tensor but a cpu tensor, and then the set_ call
                        # will fail due to device mismatch.  no_dispatch() is
                        # not enough, because the fake tensor will still claim
                        # to be a CPU tensor and you'll end up in the CPU
                        # kernel.  Arguably this is a hack; a cleaner way to
                        # solve this is to have a FakeStorage concept which
                        # would report it's CPU device--no problem now!  But
                        # this is difficult to do because we don't have storage
                        # subclasses.  Relevant test is
                        # DynamicShapesFunctionTests::test_add_dynamic_shapes in
                        # test/dynamo/test_dynamic_shapes.py
                        maybe_fake_mgr: ContextManager[None] = contextlib.nullcontext()
                        from torch._subclasses.fake_tensor import (
                            in_kernel_invocation_manager,
                            maybe_get_fake_mode,
                        )

                        mb_fake_mode = maybe_get_fake_mode(r)
                        if mb_fake_mode is not None:
                            maybe_fake_mgr = in_kernel_invocation_manager(mb_fake_mode)
                        with torch.no_grad(), maybe_suppress():
                            with maybe_fake_mgr:
                                r.set_(r_s, storage_offset, sizes, strides)
                            if self.copy_data:
                                with torch.no_grad(), no_dispatch():
                                    r.real_tensor.set_(
                                        r_s.real_storage,
                                        t.storage_offset,
                                        t.size,
                                        t.stride,
                                    )

                if t.grad is not None:
                    from torch._dynamo.source import AttrSource

                    # TODO: Use a valid grad-specific symbolic context instead of recycling
                    # the one from t. This isn't correct if e.g. t._is_view() != t.grad._is_view().
                    r.grad = self.meta_tensor(
                        t.grad,
                        shape_env,
                        callback,
                        source=AttrSource(source, "grad"),
                        symbolic_context=symbolic_context,
                    )
                torch._C._set_conj(r, t.is_conj)
                torch._C._set_neg(r, t.is_neg)
            # This can be skipped if necessary for performance reasons
            skip_leaf = (
                t.is_gradtrackingtensor and t.level == GRAD_TENSOR_SENTINEL_VALUE
            )
            assert_metadata_eq(assert_eq, t, r, skip_symbolic=True, skip_leaf=skip_leaf)
            # Thanks to storage resizing, it's possible to end up with a tensor
            # that advertises a real size, but has a storage that actually has zero bytes.
            # Need to reflect this in the generated FakeTensor.
            if t.storage is not None and t.storage.size == 0:
                r.untyped_storage().resize_(0)

            if t.is_parameter:
                r._is_param = True

            self.set_tensor_memo(t, r)

        return self.get_tensor_memo(t)

    def __call__(
        self,
        t,
        shape_env=None,
        *,
        callback=lambda t: t(),
        source=None,
        symbolic_context=None,
        # Controls whether or not we should dump the tensor metadata to structured logs
        # when source is not None.  Because we refakify after Dynamo is done,
        # we don't want to dump info again from AOTAutograd, it is redundant.
        trace=True,
    ):
        # TODO: zero tensors?  We appear to have eliminated them by
        # excluding complex for now

        # Filter out cases we don't support
        # TODO: This can probably be simplified quite a bit
        if isinstance(t, torch.Tensor) or is_traceable_wrapper_subclass(t):
            if (
                # Lazy tensors are not supported.  Note that XLA is
                # implemented on top of lazy tensor, not excluded here; we
                # have some special handling for it; this is for XLA Dynamo
                # integration
                t.device.type == "lazy"
                or
                # Quantization is not supported
                t.is_quantized
                or
                # Views out of sparse tensors not currently supported (plain
                # sparse is supported htough)
                (t._is_view() and t._base is not None and t._base.is_sparse)
            ):
                self.miss += 1
                return NotImplemented
            else:
                self.hit += 1
        elif torch.overrides.is_tensor_like(t):
            self.miss += 1
            return NotImplemented
        else:
            # non-Tensor types don't count as hit or miss
            return t

        if source is None:
            trace = False

        # Describe the tensor.  NB: do NOT disable ambient modes, we may need
        # to query them when figuring out what to put in here
        t_desc = self.describer.describe_tensor(t, trace=trace)

        if trace:
            trace_structured(
                "describe_source",
                metadata_fn=lambda: {
                    "describer_id": self.describer.id,
                    "id": t_desc.id,
                    "source": source.name(),
                },
            )

        # Do the meta-fication.  Here, we disable all the ambient modes, to
        # better simulate what would be like to re-fakeify from a fresh
        # process
        with contextlib.ExitStack() as exit_stack:
            exit_stack.enter_context(torch._dispatch.python.suspend_functionalization())
            st = peek_interpreter_stack()
            if st is not None:
                exit_stack.enter_context(
                    torch._functorch.pyfunctorch.temporarily_clear_interpreter_stack()
                )

            r = self.meta_tensor(
                t_desc,
                shape_env=shape_env,
                callback=callback,
                source=source,
                symbolic_context=symbolic_context,
            )

        if type(t) is torch.nn.Parameter:
            # NB: Cannot directly use Parameter constructor
            # because that would force a detach, not desirable
            r._is_param = True

        # TODO: return the description for later
        return r

