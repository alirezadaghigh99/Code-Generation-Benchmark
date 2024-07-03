def check_model(
    model: ModelProto | str | bytes | os.PathLike,
    full_check: bool = False,
    skip_opset_compatibility_check: bool = False,
    check_custom_domain: bool = False,
) -> None:
    """Check the consistency of a model.

    An exception will be raised if the model's ir_version is not set
    properly or is higher than checker's ir_version, or if the model
    has duplicate keys in metadata_props.

    If IR version >= 3, the model must specify opset_import.
    If IR version < 3, the model cannot have any opset_import specified.

    Args:
        model: Model to check. If model is a path, the function checks model
            path first. If the model bytes size is larger than 2GB, function
            should be called using model path.
        full_check: If True, the function also runs shape inference check.
        skip_opset_compatibility_check: If True, the function skips the check for
            opset compatibility.
        check_custom_domain: If True, the function will check all domains. Otherwise
            only check built-in domains.
    """
    # If model is a path instead of ModelProto
    if isinstance(model, (str, os.PathLike)):
        C.check_model_path(
            os.fspath(model),
            full_check,
            skip_opset_compatibility_check,
            check_custom_domain,
        )
    else:
        protobuf_string = (
            model if isinstance(model, bytes) else model.SerializeToString()
        )
        # If the protobuf is larger than 2GB,
        # remind users should use the model path to check
        if sys.getsizeof(protobuf_string) > MAXIMUM_PROTOBUF:
            raise ValueError(
                "This protobuf of onnx model is too large (>2GB). Call check_model with model path instead."
            )
        C.check_model(
            protobuf_string,
            full_check,
            skip_opset_compatibility_check,
            check_custom_domain,
        )def check_tensor(tensor: TensorProto, ctx: C.CheckerContext = DEFAULT_CONTEXT) -> None:
    _ensure_proto_type(tensor, TensorProto)
    return C.check_tensor(tensor.SerializeToString(), ctx)def check_sparse_tensor(
    sparse: SparseTensorProto, ctx: C.CheckerContext = DEFAULT_CONTEXT
) -> None:
    _ensure_proto_type(sparse, SparseTensorProto)
    C.check_sparse_tensor(sparse.SerializeToString(), ctx)