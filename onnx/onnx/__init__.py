def load_model(
    f: IO[bytes] | str | os.PathLike,
    format: _SupportedFormat | None = None,  # noqa: A002
    load_external_data: bool = True,
) -> ModelProto:
    """Loads a serialized ModelProto into memory.

    Args:
        f: can be a file-like object (has "read" function) or a string/PathLike containing a file name
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.
        load_external_data: Whether to load the external data.
            Set to True if the data is under the same directory of the model.
            If not, users need to call :func:`load_external_data_for_model`
            with directory to load external data from.

    Returns:
        Loaded in-memory ModelProto.
    """
    model = _get_serializer(format, f).deserialize_proto(_load_bytes(f), ModelProto())

    if load_external_data:
        model_filepath = _get_file_path(f)
        if model_filepath:
            base_dir = os.path.dirname(model_filepath)
            load_external_data_for_model(model, base_dir)

    return model