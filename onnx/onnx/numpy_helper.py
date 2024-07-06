def from_dict(dict_: dict[Any, Any], name: str | None = None) -> MapProto:
    """Converts a Python dictionary into a map def.

    Args:
        dict_: Python dictionary
        name: (optional) the name of the map.

    Returns:
        MapProto: the converted map def.
    """
    map_proto = MapProto()
    if name:
        map_proto.name = name
    keys = list(dict_)
    raw_key_type = np.result_type(keys[0])
    key_type = helper.np_dtype_to_tensor_dtype(raw_key_type)

    valid_key_int_types = [
        TensorProto.INT8,
        TensorProto.INT16,
        TensorProto.INT32,
        TensorProto.INT64,
        TensorProto.UINT8,
        TensorProto.UINT16,
        TensorProto.UINT32,
        TensorProto.UINT64,
    ]

    if not (
        all(
            np.result_type(key) == raw_key_type  # type: ignore[arg-type]
            for key in keys
        )
    ):
        raise TypeError(
            "The key type in the input dictionary is not the same "
            "for all keys and therefore is not valid as a map."
        )

    values = list(dict_.values())
    raw_value_type = np.result_type(values[0])
    if not all(np.result_type(val) == raw_value_type for val in values):
        raise TypeError(
            "The value type in the input dictionary is not the same "
            "for all values and therefore is not valid as a map."
        )

    value_seq = from_list(values)

    map_proto.key_type = key_type
    if key_type == TensorProto.STRING:
        map_proto.string_keys.extend(keys)
    elif key_type in valid_key_int_types:
        map_proto.keys.extend(keys)
    map_proto.values.CopyFrom(value_seq)
    return map_proto

def from_array(arr: np.ndarray, name: str | None = None) -> TensorProto:
    """Converts a numpy array to a tensor def.

    Args:
        arr: a numpy array.
        name: (optional) the name of the tensor.

    Returns:
        TensorProto: the converted tensor def.
    """
    if not isinstance(arr, (np.ndarray, np.generic)):
        raise TypeError(
            f"arr must be of type np.generic or np.ndarray, got {type(arr)}"
        )

    tensor = TensorProto()
    tensor.dims.extend(arr.shape)
    if name:
        tensor.name = name

    if arr.dtype == object:
        # Special care for strings.
        tensor.data_type = helper.np_dtype_to_tensor_dtype(arr.dtype)
        # TODO: Introduce full string support.
        # We flatten the array in case there are 2-D arrays are specified
        # We throw the error below if we have a 3-D array or some kind of other
        # object. If you want more complex shapes then follow the below instructions.
        # Unlike other types where the shape is automatically inferred from
        # nested arrays of values, the only reliable way now to feed strings
        # is to put them into a flat array then specify type astype(object)
        # (otherwise all strings may have different types depending on their length)
        # and then specify shape .reshape([x, y, z])
        flat_array = arr.flatten()
        for e in flat_array:
            if isinstance(e, str):
                tensor.string_data.append(e.encode("utf-8"))
            elif isinstance(e, np.ndarray):
                for s in e:
                    if isinstance(s, str):
                        tensor.string_data.append(s.encode("utf-8"))
                    elif isinstance(s, bytes):
                        tensor.string_data.append(s)
            elif isinstance(e, bytes):
                tensor.string_data.append(e)
            else:
                raise NotImplementedError(
                    "Unrecognized object in the object array, expect a string, or array of bytes: ",
                    str(type(e)),
                )
        return tensor

    # For numerical types, directly use numpy raw bytes.
    try:
        dtype = helper.np_dtype_to_tensor_dtype(arr.dtype)
    except KeyError as e:
        raise RuntimeError(f"Numpy data type not understood yet: {arr.dtype!r}") from e
    tensor.data_type = dtype
    tensor.raw_data = arr.tobytes()  # note: tobytes() is only after 1.9.
    if sys.byteorder == "big":
        # Convert endian from big to little
        convert_endian(tensor)

    return tensor

