def _export(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    *,
    strict: bool = True,
    preserve_module_call_signature: Tuple[str, ...] = (),
    pre_dispatch: bool = False,
    _allow_complex_guards_as_runtime_asserts: bool = False,
    _disable_forced_specializations: Optional[bool] = False,
    _is_torch_jit_trace: bool = False,
) -> ExportedProgram:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a ExportedProgram.

    Args:
        f: the `nn.Module` to trace.

        args: example positional inputs.

        kwargs: optional example keyword inputs.

        dynamic_shapes:
         An optional argument where the type should either be:
         1) a dict from argument names of ``f`` to their dynamic shape specifications,
         2) a tuple that specifies dynamic shape specifications for each input in original order.
         If you are specifying dynamism on keyword args, you will need to pass them in the order that
         is defined in the original function signature.

         The dynamic shape of a tensor argument can be specified as either
         (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
         not required to include static dimension indices in this dict, but when they are,
         they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
         where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
         are denoted by None. Arguments that are dicts or tuples / lists of tensors are
         recursively specified by using mappings or sequences of contained specifications.

        preserve_module_call_signature: A list of submodule paths for which the original
            calling conventions are preserved as metadata.

        _allow_complex_guards_as_runtime_asserts:
         With the current dynamic shapes language for dims and derived dims, we can run into constraints
         that are not expressible with the language. For example, flattening a matrix and adding to a vector,
         both fully dynamic (i.e. x.reshape([-1]) + y) emits a guard s0 * s1 = s2, which is not expressible.
         By default, we either raise a constraint violation error or specialize to static values.
         If this flag is set to True, we avoid erroring out and instead allow complex constraints to exist as runtime
         assertions in the graph. The sympy interpreter (torch/utils/_sympy/interp.py) will produce the math ops
         required to compute and assert the value of the guard (e.g. sym_size_int, eq, _assert_scalar).
         Additionally, if TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS=1 is specified, we will allow complex constraints
         while not emitting runtime asserts, returning a cleaner graph with lesser guarantees around dynamic shapes.

        _disable_forced_specializations:
         Similar to _allow_complex_guards_as_runtime_asserts, but only avoids specializing to static values if set to True.
         For complex guards that don't specialize, this flag doesn't have any effect. Ideally this would be subsumed by
         _allow_complex_guards_as_runtime_asserts, but this handles one additional case: single-variable equalities where
         the symbol is solvable for a concrete value (e.g. Eq(s0 // 4, 400) -> s0 = 1600). If set to True, this flag will
         avoid specializations. Direct equalities (e.g. s0 = 4), will still specialize.

    Returns:
        An ExportedProgram containing the traced method.
    """
    if not isinstance(args, tuple):
        raise UserError(
            UserErrorType.INVALID_INPUT,
            f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}",
        )

    if _disable_forced_specializations and strict:
        raise UserError(
            UserErrorType.INVALID_INPUT,
            "_disable_forced_specializations can be only be specified in non-strict mode.",
        )

    global _EXPORT_FLAGS, _EXPORT_MODULE_HIERARCHY
    _EXPORT_MODULE_HIERARCHY = _get_module_hierarchy(mod)

    flags = set()
    flags.add("strict" if strict else "non_strict")
    flags.add("pre_dispatch" if pre_dispatch else "aot_dispatch")
    log_export_usage(event="export.enter", flags=flags)
    _EXPORT_FLAGS = flags

    kwargs = kwargs or {}
    if isinstance(dynamic_shapes, torch.export.ShapesCollection):
        dynamic_shapes = dynamic_shapes.dynamic_shapes(mod, args, kwargs)

    flat_args, orig_in_spec = pytree.tree_flatten((args, kwargs))
    original_state_dict = mod.state_dict(keep_vars=True)
    if not _is_torch_jit_trace:
        forward_arg_names = _get_forward_arg_names(mod, args, kwargs)
    else:
        forward_arg_names = None

    # Call the appropriate export function based on the strictness of tracing.
    export_func = _strict_export if strict else _non_strict_export

    export_artifact = export_func(
        mod,
        args,
        kwargs,
        dynamic_shapes,
        preserve_module_call_signature,
        pre_dispatch,
        original_state_dict,
        orig_in_spec,
        _allow_complex_guards_as_runtime_asserts,
        _disable_forced_specializations,
        _is_torch_jit_trace,
    )
    # Decompose here for readability.
    gm = export_artifact.aten.gm
    export_graph_signature = export_artifact.aten.sig
    out_spec = export_artifact.out_spec
    fake_mode = export_artifact.fake_mode
    module_call_specs = export_artifact.module_call_specs

    # Add forward args metadata.
    gm.meta["forward_arg_names"] = forward_arg_names

    # The unbacked symint symbols are updated in aot_export
    # so we serialize them here instead of inside dynamo.
    gm.meta["inline_constraints"] = {
        k: v
        for k, v in fake_mode.shape_env.var_to_range.items()
        if free_unbacked_symbols(k)
    }
    num_lifted = next(
        (
            i
            for i, s in enumerate(export_graph_signature.input_specs)
            if s.kind == InputKind.USER_INPUT
        ),
        len(export_graph_signature.input_specs),
    )
    combined_args = _combine_args(
        mod, args, kwargs, _is_torch_jit_trace=_is_torch_jit_trace
    )
    range_constraints = make_constraints(
        fake_mode,
        gm,
        combined_args,
        dynamic_shapes,
        num_lifted,
    )
    if strict:
        _add_runtime_assertions_to_cond_in_subgraph(
            range_constraints,
            gm,
            fake_mode,
        )

    # Make module signatures.
    module_call_signatures = {}
    for fqn, specs in module_call_specs.items():
        mod_fqn = _strip_root(fqn) if not strict else fqn
        module_call_signatures[mod_fqn] = ModuleCallSignature(
            inputs=[], outputs=[], **specs
        )

    if len(preserve_module_call_signature) > 0:
        if not strict:
            _rewrite_node(gm)
        res = CollectTracepointsPass(module_call_signatures, export_graph_signature)(gm)
        assert res is not None
        gm = res.graph_module

    assert out_spec is not None

    _verify_nn_module_stack(gm)
    _verify_stack_trace(gm)
    if not _is_torch_jit_trace:
        _verify_placeholder_names(gm, export_graph_signature)

    assert _EXPORT_MODULE_HIERARCHY is not None
    exported_program = ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=export_graph_signature,
        state_dict=original_state_dict,
        range_constraints=range_constraints,
        module_call_graph=_make_module_call_graph(
            _EXPORT_MODULE_HIERARCHY,
            orig_in_spec,
            out_spec,
            module_call_signatures,
        ),
        example_inputs=(args, kwargs),
        constants=export_artifact.aten.constants,
    )

    return exported_program

