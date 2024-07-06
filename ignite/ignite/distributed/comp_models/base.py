def _encode_input_data(x: Union[torch.Tensor, float, str, None], is_src: bool) -> List[int]:
        encoded_msg = [-1] * 512
        if not is_src:
            # Discard input type if not source
            return encoded_msg

        if isinstance(x, torch.Tensor):
            shape = x.shape
            dtype = str(x.dtype)
            msg = [0, len(shape), *shape, len(dtype), *list(bytearray(dtype, "utf-8"))]
            encoded_msg[: len(msg)] = msg
        elif isinstance(x, Number):
            encoded_msg[0] = 1
        elif isinstance(x, str):
            encoded_msg[0] = 2
        return encoded_msg

