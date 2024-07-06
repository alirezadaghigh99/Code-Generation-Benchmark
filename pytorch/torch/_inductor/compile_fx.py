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

