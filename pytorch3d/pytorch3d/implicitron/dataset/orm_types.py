def TupleTypeFactory(dtype=float, shape: Tuple[int, ...] = (2,)):
    format_symbol = {
        float: "f",  # float32
        int: "i",  # int32
    }[dtype]

    class TupleType(TypeDecorator):
        impl = LargeBinary
        _format = format_symbol * math.prod(shape)

        def process_bind_param(self, value, _):
            if value is None:
                return None

            if len(shape) > 1:
                value = np.array(value, dtype=dtype).reshape(-1)

            return struct.pack(TupleType._format, *value)

        def process_result_value(self, value, _):
            if value is None:
                return None

            loaded = struct.unpack(TupleType._format, value)
            if len(shape) > 1:
                loaded = _rec_totuple(
                    np.array(loaded, dtype=dtype).reshape(shape).tolist()
                )

            return loaded

    return TupleType