def parametrize(
    *, use_const_ref_for_mutable_tensors: bool, use_ilistref_for_tensor_lists: bool
) -> Iterator[None]:
    old_use_const_ref_for_mutable_tensors = _locals.use_const_ref_for_mutable_tensors
    old_use_ilistref_for_tensor_lists = _locals.use_ilistref_for_tensor_lists
    try:
        _locals.use_const_ref_for_mutable_tensors = use_const_ref_for_mutable_tensors
        _locals.use_ilistref_for_tensor_lists = use_ilistref_for_tensor_lists
        yield
    finally:
        _locals.use_const_ref_for_mutable_tensors = (
            old_use_const_ref_for_mutable_tensors
        )
        _locals.use_ilistref_for_tensor_lists = old_use_ilistref_for_tensor_lists

