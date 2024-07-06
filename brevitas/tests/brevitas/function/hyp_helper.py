def tensor_clamp_ste_test_st():
    """
    Generate input values for testing tensor_clamp_ste fwd or bwd.
    """
    return st.one_of(
        tensor_clamp_ste_min_max_scalar_tensor_test_st(), tensor_clamp_ste_random_shape_test_st())

