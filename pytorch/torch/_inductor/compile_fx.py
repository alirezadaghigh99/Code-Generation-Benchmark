def compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    cudagraphs: Optional[BoxedBool] = None,
    static_input_idxs: Optional[List[int]] = None,
    is_backward: bool = False,
    graph_id: Optional[int] = None,
    cpp_wrapper: bool = False,
    aot_mode: bool = False,
    is_inference: bool = False,
    boxed_forward_device_index: Optional[BoxedDeviceIndex] = None,
    user_visible_outputs: Optional[Dict[str, None]] = None,
    layout_opt: Optional[bool] = None,
    extern_node_serializer: Optional[Callable[[List[ExternKernelNode]], Any]] = None,
) -> Union[CompiledFxGraph, str]:
    """
    Inductor API that compiles a single graph.

    If you change the argument list for this function, make sure you
    also update the call to save_args_for_compile_fx_inner below accordingly.
    """
    if dynamo_utils.count_calls(gm.graph) == 0 and not aot_mode:
        # trigger the real recompilation for _LazyGraphModule before returning
        # the forward method.
        from torch.fx._lazy_graph_module import _LazyGraphModule

        _LazyGraphModule.force_recompile(gm)
        return make_boxed_func(gm.forward)

    if static_input_idxs is None:
        static_input_idxs = []

    assert isinstance(
        next(iter(reversed(gm.graph.nodes))).args[0], (tuple, list)
    ), f"inductor can only compile FX graphs which return a tuple/list, but got {gm.graph}"

    if config.save_args:
        save_args_for_compile_fx_inner(
            gm,
            example_inputs,
            cudagraphs=cudagraphs,
            static_input_idxs=static_input_idxs,
            is_backward=is_backward,
            graph_id=graph_id,
            cpp_wrapper=cpp_wrapper,
            aot_mode=aot_mode,
            is_inference=is_inference,
            boxed_forward_device_index=boxed_forward_device_index,
            user_visible_outputs=user_visible_outputs,
            layout_opt=layout_opt,
        )

    if cudagraphs is None:
        cudagraphs = BoxedBool(config.triton.cudagraphs)

    # Inputs to fx_codegen_and_compile
    # Anything that affects codegen should go here, so if the signature
    # of fx_codegen_and_compile changes, the dict should be updated accordingly
    graph_kwargs = {
        "cudagraphs": cudagraphs,
        "static_input_idxs": static_input_idxs,
        "is_backward": is_backward,
        "graph_id": graph_id,
        "cpp_wrapper": cpp_wrapper,
        "aot_mode": aot_mode,
        "is_inference": is_inference,
        "user_visible_outputs": user_visible_outputs,
        "layout_opt": layout_opt,
        "extern_node_serializer": extern_node_serializer,
    }

    start = time.time()

    fx_graph_remote_cache = should_use_remote_fx_graph_cache()
    inputs_to_check = get_input_idxs_to_check(example_inputs, static_input_idxs)
    if (
        not config.force_disable_caches
        and (config.fx_graph_cache or fx_graph_remote_cache)
        and not aot_mode
    ):
        for i, input in enumerate(example_inputs):
            if (
                isinstance(input, torch.Tensor)
                and input.device.type == "cuda"
                and i in static_input_idxs
            ):
                input._is_inductor_static = True  # type: ignore[attr-defined]

        compiled_graph = FxGraphCache.load(
            fx_codegen_and_compile,
            gm,
            example_inputs,
            graph_kwargs,
            inputs_to_check,
            local=config.fx_graph_cache,
            remote=fx_graph_remote_cache,
        )
    else:
        compiled_graph = fx_codegen_and_compile(
            gm, example_inputs, **graph_kwargs  # type: ignore[arg-type]
        )

    log.debug("FX codegen and compilation took %.3fs", time.time() - start)

    # check cudagraph disabling reasons from inductor lowering
    if cudagraphs and compiled_graph.disabled_cudagraphs_reason:
        if "cuda" in compiled_graph.device_types:
            log_cudagraph_skip_and_bump_counter(
                f"skipping cudagraphs due to {compiled_graph.disabled_cudagraphs_reason}"
            )
        else:
            counters["inductor"]["cudagraph_skips"] += 1
        BoxedBool.disable(cudagraphs)

    # Return the output strides to the caller via TracingContext
    context = torch._guards.TracingContext.try_get()
    if context is not None and context.output_strides is not None:
        assert len(context.output_strides) == 0
        shape_env = _shape_env_from_inputs(example_inputs)
        for exprs in compiled_graph.output_strides:
            if exprs is None:
                context.output_strides.append(None)
            else:
                context.output_strides.append(
                    tuple(
                        (
                            shape_env.evaluate_symexpr(e)
                            if shape_env is not None
                            else int(e)
                        )
                        for e in exprs
                    )
                )

    if aot_mode:
        return compiled_graph

    if cudagraphs:
        # output args are tuple of first argument
        output = output_node(gm)
        assert len(output.args) == 1
        stack_traces = [
            (arg.stack_trace if isinstance(arg, torch.fx.node.Node) else None)
            for arg in output.args[0]
        ]

        complex_memory_overlap_inputs = any(
            complex_memory_overlap(t)
            for t in example_inputs
            if isinstance(t, torch.Tensor)
        )

        if not config.triton.cudagraph_support_input_mutation:
            # Skip supports for cudagraph-managed tensors
            from torch._inductor.cudagraph_utils import (
                check_for_mutation_ignore_cuda_graph_managed_tensor,
            )

            has_mutation_str = check_for_mutation_ignore_cuda_graph_managed_tensor(
                gm, compiled_graph, static_input_idxs
            )
            has_mutation = has_mutation_str is not None

            if has_mutation:
                compiled_graph.disabled_cudagraphs_reason = has_mutation_str
        else:
            # Check mutation later to support cudagraph-managed tensors
            has_mutation = None

        cudagraph_tests = [
            (not has_mutation, "mutated inputs"),
            (not has_incompatible_cudagraph_ops(gm), "incompatible ops"),
            (not complex_memory_overlap_inputs, "complex memory overlap"),
            (
                all(
                    isinstance(t, (torch.Tensor, torch.SymInt)) for t in example_inputs
                ),
                "non-Tensor inputs",
            ),
        ]
        cudagraph_fail_reasons = [s for b, s in cudagraph_tests if not b]

        if not cudagraph_fail_reasons:
            if not config.triton.cudagraph_trees:
                # Force specialize all inputs so that CUDA graphs will work
                for t in example_inputs:
                    if isinstance(t, torch.SymInt):
                        int(t)  # guard

            if (
                boxed_forward_device_index is not None
                and not is_inference
                and not is_backward
            ):
                boxed_forward_device_index.set(next(iter(compiled_graph.device_idxs)))

            compiled_graph.current_callable = cudagraphify(
                compiled_graph.current_callable,
                example_inputs,
                static_input_idxs=static_input_idxs,
                device_index=next(iter(compiled_graph.device_idxs)),
                stack_traces=stack_traces,
                is_backward=is_backward,
                is_inference=is_inference,
                constants=tuple(compiled_graph.constants.values()),
                placeholders=tuple(get_placeholders(gm.graph)),
                mutated_input_idxs=tuple(compiled_graph.mutated_input_idxs),
            )
        else:
            BoxedBool.disable(cudagraphs)

            # See [Backward Generation Handling]
            # if cudagraph'd the forward and set the device, we need to let the cudagraph manager
            # know we are we running the backward even if we will not run it in cudagraphs
            if is_backward and config.triton.cudagraph_trees:
                assert boxed_forward_device_index is not None
                assert boxed_forward_device_index.value is not None
                compiled_graph_callable = compiled_graph.current_callable

                manager = torch._inductor.cudagraph_trees.get_manager(
                    boxed_forward_device_index.value, create_if_none_exists=False
                )
                # should already exist from forward
                assert manager is not None

                def compiled_artifact(new_inputs):
                    manager.set_to_running_backward()  # type: ignore[union-attr]
                    return compiled_graph_callable(new_inputs)

                compiled_graph.current_callable = compiled_artifact

            if "cuda" in compiled_graph.device_types:
                # prefer better disable_cudagraphs_reason bc stack trace
                # TODO: migrate all disable reasons to stack trace, refactor
                if compiled_graph.disabled_cudagraphs_reason:
                    log_cudagraph_skip_and_bump_counter(
                        compiled_graph.disabled_cudagraphs_reason
                    )
                else:
                    log_cudagraph_skip_and_bump_counter(
                        f"skipping cudagraphs due to {cudagraph_fail_reasons}"
                    )

    # cudagraphs does its own aligning of inputs
    if not cudagraphs:
        new_callable = align_inputs_from_check_idxs(
            compiled_graph.current_callable, inputs_to_check
        )
        if new_callable is not compiled_graph.current_callable:
            compiled_graph.current_callable = new_callable

    _step_logger()(
        logging.INFO,
        "torchinductor done compiling "
        f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
        f"graph {graph_id}",
    )

    # aot autograd needs to know to pass in inputs as a list
    compiled_graph._boxed_call = True
    return compiled_graph

def compile_fx(
    model_: torch.fx.GraphModule,
    example_inputs_: List[torch.Tensor],
    inner_compile: Callable[..., Any] = compile_fx_inner,
    config_patches: Optional[Dict[str, Any]] = None,
    decompositions: Optional[Dict[OpOverload, Callable[..., Any]]] = None,
):
    """Main entrypoint to a compile given FX graph"""
    if config_patches:
        with config.patch(config_patches):
            return compile_fx(
                model_,
                example_inputs_,
                # need extra layer of patching as backwards is compiled out of scope
                inner_compile=config.patch(config_patches)(inner_compile),
                decompositions=decompositions,
            )

    if config.cpp_wrapper:
        with config.patch(
            {
                "cpp_wrapper": False,
                # For triton.autotune_at_compile_time, disable by default for
                # FBCode, but enabled by default for OSS.
                "triton.autotune_at_compile_time": config.triton.autotune_at_compile_time
                if config.is_fbcode()
                else os.environ.get(
                    "TORCHINDUCTOR_TRITON_AUTOTUNE_AT_COMPILE_TIME", "1"
                )
                == "1",
                "triton.autotune_cublasLt": False,
                "triton.cudagraphs": False,
                "triton.store_cubin": True,
            }
        ), V.set_real_inputs(example_inputs_):
            inputs_ = example_inputs_
            if isinstance(model_, torch.fx.GraphModule):
                fake_inputs = [
                    node.meta.get("val")
                    for node in model_.graph.nodes
                    if node.op == "placeholder"
                ]
                if all(v is not None for v in fake_inputs):
                    # Validate devices before switching to fake tensors.
                    for idx, fi, i in zip(count(), fake_inputs, inputs_):
                        if fi.device != i.device:
                            raise ValueError(
                                f"Device mismatch between fake input and example input at position #{idx}: "
                                f"{fi.device} vs {i.device}. If the model was exported via torch.export(), "
                                "make sure torch.export() and torch.aot_compile() run on the same device."
                            )
                    inputs_ = fake_inputs
            return compile_fx(
                model_,
                inputs_,
                inner_compile=functools.partial(inner_compile, cpp_wrapper=True),
                decompositions=decompositions,
            )

    recursive_compile_fx = functools.partial(
        compile_fx,
        inner_compile=inner_compile,
        decompositions=decompositions,
    )

    if not graph_returns_tuple(model_):
        return make_graph_return_tuple(
            model_,
            example_inputs_,
            recursive_compile_fx,
        )

    if isinstance(model_, torch.fx.GraphModule):
        if isinstance(model_.graph._codegen, _PyTreeCodeGen):
            # this graph is the result of dynamo.export()
            return handle_dynamo_export_graph(
                model_,
                example_inputs_,
                recursive_compile_fx,
            )

        model_ = _recursive_pre_grad_passes(model_, example_inputs_)

    if any(isinstance(x, (list, tuple, dict)) for x in example_inputs_):
        return flatten_graph_inputs(
            model_,
            example_inputs_,
            recursive_compile_fx,
        )

    assert not config._raise_error_for_testing
    num_example_inputs = len(example_inputs_)
    cudagraphs = BoxedBool(config.triton.cudagraphs)
    forward_device = BoxedDeviceIndex(None)

    graph_id = next(_graph_counter)

    decompositions = (
        decompositions if decompositions is not None else select_decomp_table()
    )

    @dynamo_utils.dynamo_timed
    def fw_compiler_base(
        model: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        is_inference: bool,
    ):
        if is_inference:
            # partition_fn won't be called
            _recursive_joint_graph_passes(model)

        fixed = torch._inductor.utils.num_fw_fixed_arguments(
            num_example_inputs, len(example_inputs)
        )

        user_visible_outputs = {}

        if config.keep_output_stride:
            model_outputs_node = output_node(model)
            model_outputs = pytree.arg_tree_leaves(*model_outputs_node.args)
            num_model_outputs = len(model_outputs)

            context = torch._guards.TracingContext.try_get()
            # See Note [User Outputs in the inductor graph]
            if context is not None and context.fw_metadata and not is_inference:
                original_output_start_index = (
                    context.fw_metadata.num_mutated_inp_runtime_indices
                )
            else:
                original_output_start_index = 0

            if isinstance(model_, torch.fx.GraphModule):
                *_, orig_model_outputs_node = model_.graph.nodes
                assert orig_model_outputs_node.op == "output"
                orig_model_outputs, _ = pytree.tree_flatten(
                    orig_model_outputs_node.args
                )
                num_orig_model_outputs = len(orig_model_outputs)
            else:
                num_orig_model_outputs = num_model_outputs

            assert num_orig_model_outputs <= num_model_outputs

            # Note [User Outputs in the inductor graph]
            # We makes the following assumption
            # For inference
            #   len(orig_model_outputs) == len(model_outputs)
            # For training
            #   len(orig_model_outputs) <= len(model_outputs)
            # During training, most of the time the model_outputs starts with
            # original module's outputs followed by saved activations.
            # But this can be not true if the model have inplace updated tensors.
            # AOTAutograd will make those tensors being returned before the original
            # module's output.
            # To make things safe, we'll use original_output_start_index field
            # set by AOTAutograd to decide where the original module outputs start.
            orig_output_end_idx = original_output_start_index + num_orig_model_outputs
            # Sanity chec: we are about to splice out the "user" outputs from the full set
            # of "graph" outputs. Make sure we're within bounds.
            assert orig_output_end_idx <= num_model_outputs

            user_visible_outputs = dict.fromkeys(
                n.name
                for n in model_outputs[original_output_start_index:orig_output_end_idx]
                if isinstance(n, torch.fx.Node)
            )

        return inner_compile(
            model,
            example_inputs,
            static_input_idxs=get_static_input_idxs(fixed),
            cudagraphs=cudagraphs,
            graph_id=graph_id,
            is_inference=is_inference,
            boxed_forward_device_index=forward_device,
            user_visible_outputs=user_visible_outputs,
        )

    fw_compiler = functools.partial(fw_compiler_base, is_inference=False)

    if config.freezing and not torch.is_grad_enabled():
        inference_compiler = functools.partial(
            fw_compiler_freezing,
            dynamo_model=model_,
            num_example_inputs=num_example_inputs,
            inner_compile=inner_compile,
            cudagraphs=cudagraphs,
            graph_id=graph_id,
            forward_device=forward_device,
        )
    else:
        inference_compiler = functools.partial(fw_compiler_base, is_inference=True)

    def partition_fn(graph, joint_inputs, **kwargs):
        _recursive_joint_graph_passes(graph)
        return min_cut_rematerialization_partition(
            graph, joint_inputs, **kwargs, compiler="inductor"
        )

    @compile_time_strobelight_meta(phase_name="bw_compiler")
    @dynamo_utils.dynamo_timed
    def bw_compiler(model: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        user_visible_outputs = {}

        if config.bw_outputs_user_visible:
            model_outputs_node = output_node(model)
            model_outputs = pytree.arg_tree_leaves(*model_outputs_node.args)
            user_visible_outputs = dict.fromkeys(
                n.name for n in model_outputs if isinstance(n, torch.fx.Node)
            )
        fixed = count_tangents(model)
        return inner_compile(
            model,
            example_inputs,
            static_input_idxs=list(range(fixed)),
            cudagraphs=cudagraphs,
            is_backward=True,
            graph_id=graph_id,
            boxed_forward_device_index=forward_device,
            user_visible_outputs=user_visible_outputs,
        )

    # TODO: can add logging before/after the call to create_aot_dispatcher_function
    # in torch._functorch/aot_autograd.py::aot_module_simplified::aot_function_simplified::new_func
    # once torchdynamo is merged into pytorch

    fake_mode = detect_fake_mode(example_inputs_) or torch._subclasses.FakeTensorMode(
        allow_non_fake_inputs=True
    )
    tracing_context = (
        torch._guards.TracingContext.try_get()
        or torch._guards.TracingContext(fake_mode)
    )

    if V.aot_compilation is True:
        with functorch_config.patch(unlift_effect_tokens=True):
            gm, graph_signature = aot_export_module(
                model_,
                example_inputs_,
                trace_joint=False,
                decompositions=decompositions,
            )
        unlifted_gm = _unlift_graph(model_, gm, graph_signature)
        if "dynamo_flat_name_to_original_fqn" in model_.meta:
            unlifted_gm.meta["dynamo_flat_name_to_original_fqn"] = model_.meta[
                "dynamo_flat_name_to_original_fqn"
            ]

        # Disable amp as in aot_dispatch_autograd (https://github.com/pytorch/pytorch/pull/86515)
        # In inference_compiler (fw_compiler_base), _recursive_joint_graph_passes will call into
        # _sfdp_init() to register patterns.
        # When fallback_random is set to True, the sdpa patterns will be traced during runtime.
        # If amp is turned on, the traced FP32 patterns will have prims.convert_element_type which
        # will be the same as the generated FP16 patterns.
        disable_amp = torch._C._is_any_autocast_enabled()
        context = torch._C._DisableAutocast if disable_amp else contextlib.nullcontext
        with V.set_fake_mode(fake_mode), compiled_autograd.disable(), context():
            return inference_compiler(unlifted_gm, example_inputs_)

    with V.set_fake_mode(fake_mode), torch._guards.tracing(
        tracing_context
    ), compiled_autograd.disable(), functorch_config.patch(unlift_effect_tokens=True):
        return aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            inference_compiler=inference_compiler,
            decompositions=decompositions,
            partition_fn=partition_fn,
            keep_inference_input_mutations=True,
        )(model_, example_inputs_)

