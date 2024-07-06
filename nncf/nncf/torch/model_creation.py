def create_compressed_model(
    model: Module,
    config: NNCFConfig,
    compression_state: Optional[Dict[str, Any]] = None,
    dummy_forward_fn: Callable[[Module], Any] = None,
    wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
    wrap_outputs_fn: Callable[[Any], Any] = None,
    dump_graphs=True,
) -> Tuple[CompressionAlgorithmController, NNCFNetwork]:
    """
    The main function used to produce a model ready for compression fine-tuning from an original PyTorch
    model and a configuration object.

    :param model: The original model. Should have its parameters already loaded from a checkpoint or another
        source.
    :param config: A configuration object used to determine the exact compression modifications to be applied
        to the model
    :type config: nncf.NNCFConfig
    :param compression_state: representation of the entire compression state to unambiguously restore
        the compressed model. Includes builder and controller states.
    :param dummy_forward_fn: if supplied, will be used instead of a *forward* function call to build
        the internal graph representation via tracing. Specifying this is useful when the original training pipeline
        has special formats of data loader output or has additional *forward* arguments other than input tensors.
        Otherwise, the *forward* call of the model during graph tracing will be made with mock tensors according
        to the shape specified in the config object. The dummy_forward_fn code MUST contain calls to
        nncf.nncf_model_input functions made with each compressed model input tensor in the underlying model's
        args/kwargs tuple, and these calls should be exactly the same as in the wrap_inputs_fn function code
        (see below); if dummy_forward_fn is specified, then wrap_inputs_fn also must be specified.
    :param wrap_inputs_fn: if supplied, will be used on the module's input arguments during a regular, non-dummy
        forward call before passing the inputs to the underlying compressed model. This is required if the model's
        input tensors that are important for compression are not supplied as arguments to the model's forward call
        directly, but instead are located in a container (such as list), and the model receives the container as an
        argument. wrap_inputs_fn should take as input two arguments - the tuple of positional arguments to the
        underlying model's forward call, and a dict of keyword arguments to the same. The function should wrap each
        tensor among nncf.nncf_model_input function, which is a no-operation function and marks the tensors as inputs
        to be traced by NNCF in the internal graph representation. Output is the tuple of (args, kwargs), where args
        and kwargs are the same as were supplied in input, but each tensor in the original input. Must be specified
        if dummy_forward_fn is specified.
    :param wrap_outputs_fn: same as `wrap_inputs_fn`, but applies to model outputs
    :param dump_graphs: Whether to dump the internal graph representation of the
        original and compressed models in the .dot format into the log directory.
    :return: A controller for the compression algorithm (or algorithms, in which case the controller
        is an instance of CompositeCompressionController) and the model ready for compression parameter training wrapped
        as an object of NNCFNetwork.
    """
    if isinstance(model, NNCFNetwork):
        raise nncf.InternalError(
            "The model object has already been compressed.\n"
            "NNCF for PyTorch modifies the model object in-place, and repeat calls to "
            "`nncf.torch.create_compressed_model` with the same model object passed as argument "
            "will lead to an incorrect attempt to compress the model twice.\n"
            "Make sure that the model object you are passing has not already been compressed (for "
            "instance, by testing `if isinstance(model, nncf.torch.nncf_network.NNCFNetwork)`).\n"
            "If you are encountering this in a Jupyter notebook context - make sure that when "
            "re-running cells involving `nncf.torch.create_compressed_model` the original model object "
            "is also re-created (via constructor call)."
        )

    set_debug_log_dir(config.get("log_dir", "."))

    is_legacy_model_state_dict = (
        compression_state is not None
        and BaseController.BUILDER_STATE not in compression_state
        and BaseController.CONTROLLER_STATE not in compression_state
    )
    maybe_convert_legacy_names_in_compress_state(compression_state)

    should_init = compression_state is None

    nncf_network = create_nncf_network(model, config, dummy_forward_fn, wrap_inputs_fn, wrap_outputs_fn)

    if dump_graphs and is_main_process():
        nncf_network.nncf.get_graph().visualize_graph(osp.join(config.get("log_dir", "."), "original_graph.dot"))
    builder = create_compression_algorithm_builder(config, should_init)

    is_state_loadable = not is_legacy_model_state_dict and compression_state is not None
    if is_state_loadable:
        builder.load_state(compression_state[BaseController.BUILDER_STATE])
    compressed_model = builder.apply_to(nncf_network)
    compression_ctrl = builder.build_controller(compressed_model)

    if is_state_loadable:
        compression_ctrl.load_state(compression_state[BaseController.CONTROLLER_STATE])

    compressed_model.nncf.set_compression_controller(compression_ctrl)

    # Required to ensure that the model leaving create_compressed_model has correct compressed graph.
    # In particular, this is currently required for correct functioning of RNNs.
    compressed_model.nncf.rebuild_graph()

    try:
        if is_legacy_model_state_dict:
            from nncf.torch import load_state

            state_dict_to_load = compression_state.get("state_dict", compression_state)
            load_state(compressed_model, state_dict_to_load, is_resume=True)
    finally:
        if dump_graphs and is_main_process():
            compressed_model_graph = compressed_model.nncf.get_graph()
            compressed_model_graph.visualize_graph(osp.join(config.get("log_dir", "."), "compressed_graph.dot"))

    synchronize_all_processes_in_distributed_mode()
    return compression_ctrl, compressed_model

def wrap_model(
    model: torch.nn.Module,
    example_input: Any,
    trace_parameters: bool = False,
) -> NNCFNetwork:
    """
    Wraps a PyTorch model to the NNCFNetwork class.

    This function dynamically extends the instance of PyTorch model with NNCF-enabling functionality.

    :param model: PyTorch model.
    :param example_input: An example input that will be used for model tracing. A tuple is interpreted
        as an example input of a set of non keyword arguments, and a dict as an example input of a set
        of keywords arguments.
    :param trace_parameters: Whether to trace model parameters. Default is False.
    :return: A model wrapped by NNCFNetwork.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"The provided model type {type(model)} is incompatible. "
            "Only models inheriting from torch.nn.Module are supported."
        )

    input_info = ExampleInputInfo.from_example_input(example_input)

    with training_mode_switcher(model, is_training=False):
        nncf_network = NNCFNetwork(
            model, input_info=input_info, replace_modules=not trace_parameters, trace_parameters=trace_parameters
        )
        nncf_network.nncf.get_tracing_context().disable_trace_dynamic_graph()

    return nncf_network

