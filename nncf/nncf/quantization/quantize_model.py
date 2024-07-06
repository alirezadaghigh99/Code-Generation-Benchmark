def compress_weights(
    model: TModel,
    mode=CompressWeightsMode.INT8_ASYM,
    ratio: Optional[float] = None,
    group_size: Optional[int] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    all_layers: Optional[bool] = None,
    dataset: Optional[Dataset] = None,
    sensitivity_metric: Optional[SensitivityMetric] = None,
    *,
    subset_size: Optional[int] = 128,
    awq: Optional[bool] = None,
    scale_estimation: Optional[bool] = None,
    gptq: Optional[bool] = None,
    advanced_parameters: Optional[AdvancedCompressionParameters] = None,
) -> TModel:
    """
    Compress model weights.

    :param model: A model to be compressed.
    :type model: TModel
    :param mode: Defines a mode for weight compression.
        INT8_SYM stands for 8-bit integer symmetric quantization of all weights without zero point.
        INT8_ASYM is the same as INT8_SYM mode, but weights are quantized to a primary precision asymmetrically
            with a typical non-fixed zero point.
        INT4_SYM stands for a mixed-precision weights quantization with 4-bit integer as a primary precision.
            Weights are quantized to a primary precision symmetrically without zero point.
            All embeddings and the last layer are always compressed to a backup precision, which is INT8_ASYM,
            by default. All others are quantized whether to 4-bit integer or to a backup precision depending on
            criteria and the given ratio.
        INT4_ASYM is the same as INT4_SYM mode, but weights are quantized to a primary precision asymmetrically
            with a typical non-fixed zero point.
        NF4 is the same as INT4_SYM mode, but primary precision is NF4 data type without zero point.
        E2M1 is the same as INT4_SYM mode, but primary precision is E2M1 data type without zero point.
    :type mode: nncf.CompressWeightsMode
    :param ratio: the ratio between baseline and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
        and the rest to INT8_ASYM).
    :type ratio: float
    :param group_size: number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping.
    :type group_size: int
    :param ignored_scope: An ignored scope that defined the list of model control
        flow graph nodes to be ignored during quantization.
    :type ignored_scope: nncf.IgnoredScope
    :param all_layers: Indicates whether embeddings and last MatMul layers should be compressed to a primary
        precision. By default, the backup precision is assigned for the embeddings and last MatMul layers.
    :type all_layers: bool
    :param dataset: Dataset used for assigning different quantization precision by finding outliers in activations.
    :type dataset: nncf.Dataset
    :param sensitivity_metric: The sensitivity metric for assigning quantization precision to layers. In order to
        preserve the accuracy of the model, the more sensitive layers receives a higher precision.
    :type sensitivity_metric: nncf.SensitivityMetric
    :param subset_size: Number of data samples to calculate activation statistics used for assigning different
        quantization precision. Defaults to 128.
    :type subset_size: int
    :param awq: Indicates whether use AWQ weights correction.
    :type awq: bool
    :param scale_estimation: Indicates whether a scale estimation algorithm is used that minimizes the L2 error
        between the original and compressed layers.
    :type scale_estimation: bool
    :param gptq: Indicates whether use GPTQ algorithm.
    :type gptq: bool
    :param advanced_parameters: Advanced parameters for compression algorithms.
    :type advanced_parameters: nncf.AdvancedCompressionParameters
    :return: The non-trainable model with compressed weights.
    """
    if mode == CompressWeightsMode.INT8:
        warning_deprecated(
            "`CompressWeightsMode.INT8` is deprecated. Please, use `CompressWeightsMode.INT8_ASYM` as value instead."
        )
        mode = CompressWeightsMode.INT8_ASYM

    backend = get_backend(model)
    compression_weights_impl = None

    if backend == BackendType.TORCH:
        from nncf.torch.model_creation import is_wrapped_model
        from nncf.torch.model_creation import wrap_model
        from nncf.torch.quantization.quantize_model import compress_weights_impl as pt_compression_weights_impl

        if mode not in [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT8_SYM]:
            raise AttributeError(
                "Torch backend supports only INT8_ASYM, INT8_SYM modes for weight compression, "
                f"but given {mode.value} mode."
            )

        if True in [awq, scale_estimation, gptq]:
            raise AttributeError(
                "Torch backend doesn`t supports scale estimation and AWQ algorithm, "
                "but awq=True or scale_estimation=True or gptq=True is specified."
            )

        if is_wrapped_model(model):
            if not model.nncf.trace_parameters:
                raise ValueError(
                    "Tracing capabilities with tracing parameters are required in the PyTorch model "
                    "for nncf.compress_weights(). Please wrap the model using "
                    "nncf.torch.wrap_model(model, example_input, trace_parameters=True) before calling "
                    "nncf.compress_weights()."
                )
        elif dataset is None:
            raise AttributeError("Please provide a dataset of at least one element for PyTorch model tracing.")
        else:
            example_input = next(iter(dataset.get_inference_data()))
            model = wrap_model(model, example_input=example_input, trace_parameters=True)
        dataset = None
        compression_weights_impl = pt_compression_weights_impl

    if backend == BackendType.OPENVINO:
        from nncf.openvino.quantization.quantize_model import compress_weights_impl as ov_compress_weights_impl

        if any((awq, scale_estimation)) and (
            dataset is None or mode in [CompressWeightsMode.NF4, CompressWeightsMode.E2M1] or group_size == -1
        ):
            raise AttributeError(
                "Scale estimation or AWQ algorithm defined, but dataset is None or mode is NF4 or group_size < 0."
            )
        if gptq and (dataset is None or group_size == -1 or mode == CompressWeightsMode.E2M1):
            raise AttributeError("GPTQ algorithm defined, but dataset is None or group_size < 0 or mode is E2M1.")

        if gptq and scale_estimation:
            raise AttributeError(
                "Simultaneous use of Scale estimation and GPTQ algorithms is not supported. Select one of them."
            )

        compression_weights_impl = ov_compress_weights_impl

    if mode in [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT8_SYM]:
        if ratio is None:
            ratio = 1
        if group_size is None:
            group_size = -1
        if ratio != 1 or group_size != -1:
            raise AttributeError(
                "INT8 mode assumes per-channel quantization of all layers in 8 bit. "
                "Default values of `ratio` (1) and `group_size` (-1) parameters can not be overridden"
            )
        options = [all_layers, sensitivity_metric, dataset, awq, scale_estimation, gptq]
        if any(option is not None for option in options):
            raise AttributeError(
                "INT8 modes do not support `all_layers`, `sensitivity_metric`, `awq`, `scale_estimation`, `gptq` "
                "and `dataset` options. Set them to None."
            )

    if ratio is None:
        ratio = 1
    if group_size is None:
        group_size = 128
    if all_layers is None:
        all_layers = False
    if awq is None:
        awq = False
    if scale_estimation is None:
        scale_estimation = False
    if gptq is None:
        gptq = False
    if ignored_scope is None:
        ignored_scope = IgnoredScope()
    if sensitivity_metric is None:
        sensitivity_metric = (
            SensitivityMetric.WEIGHT_QUANTIZATION_ERROR
            if dataset is None
            else SensitivityMetric.MAX_ACTIVATION_VARIANCE
        )
    if ratio != 1 and dataset is None and sensitivity_metric != SensitivityMetric.WEIGHT_QUANTIZATION_ERROR:
        raise AttributeError(
            f"Mixed precision selection based on the given sensitivity metric={sensitivity_metric.value} requires "
            "a dataset, but it's not provided."
        )
    if ratio < 0 or ratio > 1:
        raise ValueError(f"The ratio should be between 0 and 1, but ratio={ratio} is specified.")
    if subset_size is None or subset_size <= 0:
        raise ValueError(f"The subset_size value should be positive, but subset_size={subset_size} is given.")

    if compression_weights_impl is None:
        raise nncf.UnsupportedBackendError(f"Unsupported type of backend: {backend}")

    return compression_weights_impl(
        model,
        dataset,
        mode,
        ratio,
        group_size,
        ignored_scope,
        all_layers,
        sensitivity_metric,
        awq,
        subset_size,
        scale_estimation,
        gptq,
        advanced_parameters,
    )

