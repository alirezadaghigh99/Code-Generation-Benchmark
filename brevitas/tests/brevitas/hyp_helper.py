def float_tensor_random_size_st(
        draw, dims: int = 1, max_size: int = 3, min_val=None, max_val=None, width=FP32_BIT_WIDTH):
    """
    Generate a float tensor of a fixed number of dimensions each of random size.
    """
    shape = draw(random_tensor_shape_st(dims, dims, max_size))
    t = draw(float_tensor_st(min_val=min_val, max_val=max_val, shape=shape, width=width))
    return t

def random_minifloat_format(draw, min_bit_width=MIN_INT_BIT_WIDTH, max_bit_with=MAX_INT_BIT_WIDTH):
    """"
    Generate a minifloat format. Returns bit_width, exponent, mantissa, and signed.
    """
    # TODO: add support for new minifloat format that comes with FloatQuantTensor
    bit_width = draw(st.integers(min_value=min_bit_width, max_value=max_bit_with))
    exponent_bit_width = draw(st.integers(min_value=0, max_value=bit_width))
    signed = draw(st.booleans())
    # if no budget is left, return
    if bit_width == exponent_bit_width:
        return bit_width, exponent_bit_width, 0, False
    elif bit_width == (exponent_bit_width + int(signed)):
        return bit_width, exponent_bit_width, 0, signed
    mantissa_bit_width = bit_width - exponent_bit_width - int(signed)

    return bit_width, exponent_bit_width, mantissa_bit_width, signed

def two_float_tensor_random_shape_st(
        draw, min_dims=1, max_dims=4, max_size=3, width=FP32_BIT_WIDTH):
    """
    Generate a tuple of float tensors of the same random shape.
    """
    shape = draw(random_tensor_shape_st(min_dims, max_dims, max_size))
    size = reduce(mul, shape, 1)
    float_list1 = draw(st.lists(float_st(width=width), min_size=size, max_size=size))
    float_list2 = draw(st.lists(float_st(width=width), min_size=size, max_size=size))
    tensor1 = torch.tensor(float_list1).view(shape)
    tensor2 = torch.tensor(float_list2).view(shape)
    return tensor1, tensor2

