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