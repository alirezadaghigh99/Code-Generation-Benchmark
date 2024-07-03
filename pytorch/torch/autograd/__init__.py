def grad(
    outputs: _TensorOrTensors,
    inputs: _TensorOrTensorsOrGradEdge,
    grad_outputs: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: Optional[bool] = None,
    is_grads_batched: bool = False,
    materialize_grads: bool = False,
) -> Tuple[torch.Tensor, ...]:
    r"""Compute and return the sum of gradients of outputs with respect to the inputs.

    ``grad_outputs`` should be a sequence of length matching ``output``
    containing the "vector" in vector-Jacobian product, usually the pre-computed
    gradients w.r.t. each of the outputs. If an output doesn't require_grad,
    then the gradient can be ``None``).

    .. note::

        If you run any forward ops, create ``grad_outputs``, and/or call ``grad``
        in a user-specified CUDA stream context, see
        :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

    .. note::

        ``only_inputs`` argument is deprecated and is ignored now (defaults to ``True``).
        To accumulate gradient for other parts of the graph, please use
        ``torch.autograd.backward``.

    Args:
        outputs (sequence of Tensor): outputs of the differentiated function.
        inputs (sequence of Tensor or GradientEdge): Inputs w.r.t. which the gradient will be
            returned (and not accumulated into ``.grad``).
        grad_outputs (sequence of Tensor): The "vector" in the vector-Jacobian product.
            Usually gradients w.r.t. each output. None values can be specified for scalar
            Tensors or ones that don't require grad. If a None value would be acceptable
            for all grad_tensors, then this argument is optional. Default: None.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Default: ``False``.
        allow_unused (Optional[bool], optional): If ``False``, specifying inputs
            that were not used when computing outputs (and therefore their grad is
            always zero) is an error. Defaults to the value of ``materialize_grads``.
        is_grads_batched (bool, optional): If ``True``, the first dimension of each
            tensor in ``grad_outputs`` will be interpreted as the batch dimension.
            Instead of computing a single vector-Jacobian product, we compute a
            batch of vector-Jacobian products for each "vector" in the batch.
            We use the vmap prototype feature as the backend to vectorize calls
            to the autograd engine so that this computation can be performed in a
            single call. This should lead to performance improvements when compared
            to manually looping and performing backward multiple times. Note that
            due to this feature being experimental, there may be performance
            cliffs. Please use ``torch._C._debug_only_display_vmap_fallback_warnings(True)``
            to show any performance warnings and file an issue on github if warnings exist
            for your use case. Defaults to ``False``.
        materialize_grads (bool, optional): If ``True``, set the gradient for unused inputs
            to zero instead of None. This is useful when computing higher-order derivatives.
            If ``materialize_grads`` is ``True`` and ``allow_unused`` is ``False``, an error
            will be raised. Defaults to ``False``.

    """
    if materialize_grads and allow_unused is False:
        raise ValueError(
            "Expected allow_unused to be True or not passed when materialize_grads=True, "
            "but got: allow_unused=False."
        )
    if allow_unused is None:
        allow_unused = materialize_grads
    t_outputs = cast(
        Tuple[torch.Tensor, ...],
        (outputs,) if is_tensor_like(outputs) else tuple(outputs),
    )
    if is_tensor_like(inputs) or isinstance(inputs, graph.GradientEdge):
        inputs = cast(_TensorOrTensorsOrGradEdge, (inputs,))
    else:
        inputs = tuple(inputs)
    t_inputs = tuple(i for i in inputs if is_tensor_like(i))
    overridable_args = t_outputs + t_inputs
    if has_torch_function(overridable_args):
        return handle_torch_function(
            grad,
            overridable_args,
            t_outputs,
            inputs,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
            only_inputs=only_inputs,
            allow_unused=allow_unused,
            is_grads_batched=is_grads_batched,
            materialize_grads=materialize_grads,
        )

    if not only_inputs:
        warnings.warn(
            "only_inputs argument is deprecated and is ignored now "
            "(defaults to True). To accumulate gradient for other "
            "parts of the graph, please use torch.autograd.backward.",
            FutureWarning,
            stacklevel=2,
        )

    grad_outputs_ = _tensor_or_tensors_to_tuple(grad_outputs, len(t_outputs))
    grad_outputs_ = _make_grads(
        t_outputs, grad_outputs_, is_grads_batched=is_grads_batched
    )

    if retain_graph is None:
        retain_graph = create_graph

    # The reason we repeat the same comment several times below is because
    # some Python versions print out the first line of multi-line function
    # calls in the traceback and some print out the last line
    if is_grads_batched:

        def vjp(gO):
            return _engine_run_backward(
                t_outputs,
                gO,
                retain_graph,
                create_graph,
                inputs,
                allow_unused,
                accumulate_grad=False,
            )

        result = _vmap_internals._vmap(vjp, 0, 0, allow_none_pass_through=True)(
            grad_outputs_
        )
    else:
        result = _engine_run_backward(
            t_outputs,
            grad_outputs_,
            retain_graph,
            create_graph,
            inputs,
            allow_unused,
            accumulate_grad=False,
        )
    if materialize_grads:
        if any(
            result[i] is None and not is_tensor_like(inputs[i])
            for i in range(len(inputs))
        ):
            raise RuntimeError(
                "materialize_grads cannot be used when the given input is a GradientEdge"
            )
        result = tuple(
            output
            if output is not None
            else torch.zeros_like(input, requires_grad=True)
            for (output, input) in zip(result, inputs)
        )
    return resultdef grad(
    outputs: _TensorOrTensors,
    inputs: _TensorOrTensorsOrGradEdge,
    grad_outputs: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: Optional[bool] = None,
    is_grads_batched: bool = False,
    materialize_grads: bool = False,
) -> Tuple[torch.Tensor, ...]:
    r"""Compute and return the sum of gradients of outputs with respect to the inputs.

    ``grad_outputs`` should be a sequence of length matching ``output``
    containing the "vector" in vector-Jacobian product, usually the pre-computed
    gradients w.r.t. each of the outputs. If an output doesn't require_grad,
    then the gradient can be ``None``).

    .. note::

        If you run any forward ops, create ``grad_outputs``, and/or call ``grad``
        in a user-specified CUDA stream context, see
        :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

    .. note::

        ``only_inputs`` argument is deprecated and is ignored now (defaults to ``True``).
        To accumulate gradient for other parts of the graph, please use
        ``torch.autograd.backward``.

    Args:
        outputs (sequence of Tensor): outputs of the differentiated function.
        inputs (sequence of Tensor or GradientEdge): Inputs w.r.t. which the gradient will be
            returned (and not accumulated into ``.grad``).
        grad_outputs (sequence of Tensor): The "vector" in the vector-Jacobian product.
            Usually gradients w.r.t. each output. None values can be specified for scalar
            Tensors or ones that don't require grad. If a None value would be acceptable
            for all grad_tensors, then this argument is optional. Default: None.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Default: ``False``.
        allow_unused (Optional[bool], optional): If ``False``, specifying inputs
            that were not used when computing outputs (and therefore their grad is
            always zero) is an error. Defaults to the value of ``materialize_grads``.
        is_grads_batched (bool, optional): If ``True``, the first dimension of each
            tensor in ``grad_outputs`` will be interpreted as the batch dimension.
            Instead of computing a single vector-Jacobian product, we compute a
            batch of vector-Jacobian products for each "vector" in the batch.
            We use the vmap prototype feature as the backend to vectorize calls
            to the autograd engine so that this computation can be performed in a
            single call. This should lead to performance improvements when compared
            to manually looping and performing backward multiple times. Note that
            due to this feature being experimental, there may be performance
            cliffs. Please use ``torch._C._debug_only_display_vmap_fallback_warnings(True)``
            to show any performance warnings and file an issue on github if warnings exist
            for your use case. Defaults to ``False``.
        materialize_grads (bool, optional): If ``True``, set the gradient for unused inputs
            to zero instead of None. This is useful when computing higher-order derivatives.
            If ``materialize_grads`` is ``True`` and ``allow_unused`` is ``False``, an error
            will be raised. Defaults to ``False``.

    """
    if materialize_grads and allow_unused is False:
        raise ValueError(
            "Expected allow_unused to be True or not passed when materialize_grads=True, "
            "but got: allow_unused=False."
        )
    if allow_unused is None:
        allow_unused = materialize_grads
    t_outputs = cast(
        Tuple[torch.Tensor, ...],
        (outputs,) if is_tensor_like(outputs) else tuple(outputs),
    )
    if is_tensor_like(inputs) or isinstance(inputs, graph.GradientEdge):
        inputs = cast(_TensorOrTensorsOrGradEdge, (inputs,))
    else:
        inputs = tuple(inputs)
    t_inputs = tuple(i for i in inputs if is_tensor_like(i))
    overridable_args = t_outputs + t_inputs
    if has_torch_function(overridable_args):
        return handle_torch_function(
            grad,
            overridable_args,
            t_outputs,
            inputs,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
            only_inputs=only_inputs,
            allow_unused=allow_unused,
            is_grads_batched=is_grads_batched,
            materialize_grads=materialize_grads,
        )

    if not only_inputs:
        warnings.warn(
            "only_inputs argument is deprecated and is ignored now "
            "(defaults to True). To accumulate gradient for other "
            "parts of the graph, please use torch.autograd.backward.",
            FutureWarning,
            stacklevel=2,
        )

    grad_outputs_ = _tensor_or_tensors_to_tuple(grad_outputs, len(t_outputs))
    grad_outputs_ = _make_grads(
        t_outputs, grad_outputs_, is_grads_batched=is_grads_batched
    )

    if retain_graph is None:
        retain_graph = create_graph

    # The reason we repeat the same comment several times below is because
    # some Python versions print out the first line of multi-line function
    # calls in the traceback and some print out the last line
    if is_grads_batched:

        def vjp(gO):
            return _engine_run_backward(
                t_outputs,
                gO,
                retain_graph,
                create_graph,
                inputs,
                allow_unused,
                accumulate_grad=False,
            )

        result = _vmap_internals._vmap(vjp, 0, 0, allow_none_pass_through=True)(
            grad_outputs_
        )
    else:
        result = _engine_run_backward(
            t_outputs,
            grad_outputs_,
            retain_graph,
            create_graph,
            inputs,
            allow_unused,
            accumulate_grad=False,
        )
    if materialize_grads:
        if any(
            result[i] is None and not is_tensor_like(inputs[i])
            for i in range(len(inputs))
        ):
            raise RuntimeError(
                "materialize_grads cannot be used when the given input is a GradientEdge"
            )
        result = tuple(
            output
            if output is not None
            else torch.zeros_like(input, requires_grad=True)
            for (output, input) in zip(result, inputs)
        )
    return result