def make_tensor(
    name: str, data_type: int, dims: Sequence[int], vals: Any, raw: bool = False
) -> TensorProto:
    """Make a TensorProto with specified arguments.  If raw is False, this
    function will choose the corresponding proto field to store the
    values based on data_type. If raw is True, use "raw_data" proto
    field to store the values, and values should be of type bytes in
    this case.

    Args:
        name (string): tensor name
        data_type (int): a value such as onnx.TensorProto.FLOAT
        dims (List[int]): shape
        vals: values
        raw (bool): if True, vals contains the serialized content of the tensor,
            otherwise, vals should be a list of values of the type defined by *data_type*

    Returns:
        TensorProto
    """
    tensor = TensorProto()
    tensor.data_type = data_type
    tensor.name = name

    if data_type == TensorProto.STRING and raw:
        raise TypeError("Can not use raw_data to store string type.")

    np_dtype = tensor_dtype_to_np_dtype(data_type)

    # Check number of vals specified equals tensor size
    expected_size = 1
    if raw:
        # NumPy doesn't have BFLOAT16. TENSOR_TYPE_MAP maps it to float32, which has the wrong itemsize.
        if data_type == TensorProto.BFLOAT16:
            expected_size = 2
        elif data_type in (
            TensorProto.FLOAT8E4M3FN,
            TensorProto.FLOAT8E4M3FNUZ,
            TensorProto.FLOAT8E5M2,
            TensorProto.FLOAT8E5M2FNUZ,
        ):
            expected_size = 1
        # NumPy doesn't have INT4. It is packed in couples to UINT8 buffers.
        elif data_type in (TensorProto.UINT4, TensorProto.INT4):
            expected_size = 0.5  # type: ignore[assignment]
        else:
            expected_size = np_dtype.itemsize

    if type(vals) is np.ndarray and len(vals.shape) > 1:
        vals = vals.flatten()
    for d in dims:
        expected_size *= d

    if len(vals) != expected_size:
        # padding of half a byte is acceptable for 4bit types
        if not (
            data_type in (TensorProto.UINT4, TensorProto.INT4)
            and len(vals) == expected_size + 0.5
        ):
            raise ValueError(
                f"Number of values does not match tensor's size. Expected {expected_size}, but it is {len(vals)}. "
            )

    if raw:
        tensor.raw_data = vals
    else:
        if data_type in (TensorProto.COMPLEX64, TensorProto.COMPLEX128):
            vals = split_complex_to_pairs(vals)
        elif data_type == TensorProto.FLOAT16:
            vals = (
                np.array(vals).astype(np_dtype).view(dtype=np.uint16).flatten().tolist()
            )
        elif data_type in (
            TensorProto.BFLOAT16,
            TensorProto.FLOAT8E4M3FN,
            TensorProto.FLOAT8E4M3FNUZ,
            TensorProto.FLOAT8E5M2,
            TensorProto.FLOAT8E5M2FNUZ,
        ):
            fcast = {
                TensorProto.BFLOAT16: float32_to_bfloat16,
                TensorProto.FLOAT8E4M3FN: float32_to_float8e4m3,
                TensorProto.FLOAT8E4M3FNUZ: lambda *args: float32_to_float8e4m3(  # type: ignore[misc]
                    *args, uz=True
                ),
                TensorProto.FLOAT8E5M2: float32_to_float8e5m2,
                TensorProto.FLOAT8E5M2FNUZ: lambda *args: float32_to_float8e5m2(  # type: ignore[misc]
                    *args, fn=True, uz=True
                ),
            }[
                data_type  # type: ignore[index]
            ]
            vals = list(
                map(  # type: ignore[call-overload]
                    fcast,
                    np.array(vals).astype(np_dtype).flatten().tolist(),
                )
            )
        elif data_type in (
            TensorProto.UINT4,
            TensorProto.INT4,
        ):
            signed = data_type == TensorProto.INT4

            # Two packed 4-bit values must be represented as a single uint8 value.
            # Therefore, pack_float32_to_4bit() sets the dtype of the output vals
            # to uint8 regardless of the value of 'signed'. Using int8 would cause
            # the size of int4 tensors to increase ~5x if the tensor contains negative values (due to
            # the way negative values are serialized by protobuf).
            vals = pack_float32_to_4bit(vals, signed=signed).flatten().tolist()
        elif data_type == TensorProto.BOOL:
            vals = np.array(vals).astype(int)
        elif data_type == TensorProto.STRING:
            vals = np.array(vals).astype(bytes)
        field = tensor_dtype_to_field(data_type)
        getattr(tensor, field).extend(vals)
    tensor.dims.extend(dims)
    return tensordef make_node(
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    name: str | None = None,
    doc_string: str | None = None,
    domain: str | None = None,
    overload: str | None = None,
    **kwargs: Any,
) -> NodeProto:
    """Construct a NodeProto.

    Args:
        op_type (string): The name of the operator to construct
        inputs (list of string): list of input names
        outputs (list of string): list of output names
        name (string, default None): optional unique identifier for NodeProto
        doc_string (string, default None): optional documentation string for NodeProto
        domain (string, default None): optional domain for NodeProto.
            If it's None, we will just use default domain (which is empty)
        overload (string, default None): optional field, used to
            resolve calls to model-local functions
        **kwargs (dict): the attributes of the node.  The acceptable values
            are documented in :func:`make_attribute`.

    Returns:
        NodeProto
    """
    node = NodeProto()
    node.op_type = op_type
    node.input.extend(inputs)
    node.output.extend(outputs)
    if name:
        node.name = name
    if doc_string:
        node.doc_string = doc_string
    if domain is not None:
        node.domain = domain
    if overload is not None:
        node.overload = overload
    if kwargs:
        node.attribute.extend(
            make_attribute(key, value)
            for key, value in sorted(kwargs.items())
            if value is not None
        )
    return nodedef make_tensor_value_info(
    name: str,
    elem_type: int,
    shape: Sequence[str | int | None] | None,
    doc_string: str = "",
    shape_denotation: list[str] | None = None,
) -> ValueInfoProto:
    """Makes a ValueInfoProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    tensor_type_proto = make_tensor_type_proto(elem_type, shape, shape_denotation)
    value_info_proto.type.CopyFrom(tensor_type_proto)
    return value_info_protodef make_attribute(
    key: str,
    value: Any,
    doc_string: str | None = None,
    attr_type: int | None = None,
) -> AttributeProto:
    """Makes an AttributeProto based on the value type."""
    attr = AttributeProto()
    attr.name = key
    if doc_string:
        attr.doc_string = doc_string

    # Singular cases
    if isinstance(value, numbers.Integral):
        attr.i = int(value)
        attr.type = AttributeProto.INT
    elif isinstance(value, numbers.Real):
        attr.f = float(value)
        attr.type = AttributeProto.FLOAT
    elif isinstance(value, (str, bytes)):
        # Encode strings into utf-8
        attr.s = _to_bytes(value)
        attr.type = AttributeProto.STRING
    elif isinstance(value, TensorProto):
        attr.t.CopyFrom(value)
        attr.type = AttributeProto.TENSOR
    elif isinstance(value, SparseTensorProto):
        attr.sparse_tensor.CopyFrom(value)
        attr.type = AttributeProto.SPARSE_TENSOR
    elif isinstance(value, GraphProto):
        attr.g.CopyFrom(value)
        attr.type = AttributeProto.GRAPH
    elif isinstance(value, TypeProto):
        attr.tp.CopyFrom(value)
        attr.type = AttributeProto.TYPE_PROTO
    # Iterable cases
    elif isinstance(value, collections.abc.Iterable):
        value = list(value)
        if len(value) == 0 and attr_type is None:
            raise ValueError(
                f"Could not infer attribute `{key}` type from empty iterator"
            )
        if attr_type is None:
            types = {type(v) for v in value}
            for exp_t, exp_enum in (
                (numbers.Integral, AttributeProto.INTS),
                (numbers.Real, AttributeProto.FLOATS),
                ((str, bytes), AttributeProto.STRINGS),
                (TensorProto, AttributeProto.TENSORS),
                (SparseTensorProto, AttributeProto.SPARSE_TENSORS),
                (GraphProto, AttributeProto.GRAPHS),
                (TypeProto, AttributeProto.TYPE_PROTOS),
            ):
                if all(issubclass(t, exp_t) for t in types):  # type: ignore[arg-type]
                    attr_type = exp_enum
                    break
            if attr_type is None:
                raise ValueError(
                    "Could not infer the attribute type from the elements of the passed Iterable value."
                )

        if attr_type == AttributeProto.INTS:
            attr.ints.extend(value)
            attr.type = AttributeProto.INTS
        elif attr_type == AttributeProto.FLOATS:
            attr.floats.extend(value)
            attr.type = AttributeProto.FLOATS
        elif attr_type == AttributeProto.STRINGS:
            attr.strings.extend(_to_bytes(v) for v in value)
            attr.type = AttributeProto.STRINGS
        elif attr_type == AttributeProto.TENSORS:
            attr.tensors.extend(value)
            attr.type = AttributeProto.TENSORS
        elif attr_type == AttributeProto.SPARSE_TENSORS:
            attr.sparse_tensors.extend(value)
            attr.type = AttributeProto.SPARSE_TENSORS
        elif attr_type == AttributeProto.GRAPHS:
            attr.graphs.extend(value)
            attr.type = AttributeProto.GRAPHS
        elif attr_type == AttributeProto.TYPE_PROTOS:
            attr.type_protos.extend(value)
            attr.type = AttributeProto.TYPE_PROTOS
        else:
            raise AssertionError()  # Should not reach since `ValueError` must be raised in attr_type checking
    else:
        raise TypeError(f"'{value}' is not an accepted attribute value.")

    if attr_type is not None and attr.type != attr_type:
        raise TypeError(
            f"Inferred attribute type '{_attr_type_to_str(attr.type)}'({attr.type}) mismatched with specified type '{_attr_type_to_str(attr_type)}'({attr_type})"
        )
    return attrdef make_operatorsetid(
    domain: str,
    version: int,
) -> OperatorSetIdProto:
    """Construct an OperatorSetIdProto.

    Args:
        domain (string): The domain of the operator set id
        version (integer): Version of operator set id
    Returns:
        OperatorSetIdProto
    """
    operatorsetid = OperatorSetIdProto()
    operatorsetid.domain = domain
    operatorsetid.version = version
    return operatorsetiddef tensor_dtype_to_np_dtype(tensor_dtype: int) -> np.dtype:
    """Convert a TensorProto's data_type to corresponding numpy dtype. It can be used while making tensor.

    Args:
        tensor_dtype: TensorProto's data_type

    Returns:
        numpy's data_type
    """
    return mapping.TENSOR_TYPE_MAP[tensor_dtype].np_dtypedef tensor_dtype_to_storage_tensor_dtype(tensor_dtype: int) -> int:
    """Convert a TensorProto's data_type to corresponding data_type for storage.

    Args:
        tensor_dtype: TensorProto's data_type

    Returns:
        data_type for storage
    """
    return mapping.TENSOR_TYPE_MAP[tensor_dtype].storage_dtypedef tensor_dtype_to_field(tensor_dtype: int) -> str:
    """Convert a TensorProto's data_type to corresponding field name for storage. It can be used while making tensors.

    Args:
        tensor_dtype: TensorProto's data_type

    Returns:
        field name
    """
    return mapping._STORAGE_TENSOR_TYPE_TO_FIELD[
        mapping.TENSOR_TYPE_MAP[tensor_dtype].storage_dtype
    ]def _attr_type_to_str(attr_type: int) -> str:
    """Convert AttributeProto type to string.

    Args:
        attr_type: AttributeProto type.

    Returns:
        String representing the supplied attr_type.
    """
    if attr_type in AttributeProto.AttributeType.values():
        return _ATTRIBUTE_TYPE_TO_STR[attr_type]  # type: ignore[no-any-return]
    return AttributeProto.AttributeType.keys()[0]  # type: ignore[no-any-return]