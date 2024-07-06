def tensor_clamp_ste_test_st():
    """
    Generate input values for testing tensor_clamp_ste fwd or bwd.
    """
    return st.one_of(
        tensor_clamp_ste_min_max_scalar_tensor_test_st(), tensor_clamp_ste_random_shape_test_st())

def scalar_clamp_min_ste_test_st(draw):
    """
    Generate min_val float, val and val_grad tensors.
    The val and val_grad tensors has the same random shape.
    """
    shape = draw(random_tensor_shape_st())
    min_val = draw(float_st())
    val = draw(float_tensor_st(shape))
    val_grad = draw(float_tensor_st(shape))
    return min_val, val, val_grad

