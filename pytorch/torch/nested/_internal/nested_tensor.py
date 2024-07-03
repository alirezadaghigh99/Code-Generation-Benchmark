def nested_view_from_values_offsets(
    values, offsets, ragged_idx=1, min_seqlen=None, max_seqlen=None
):
    min_seqlen_tensor = None
    if min_seqlen is not None:
        min_seqlen_tensor = _store_val_in_tensor(min_seqlen)

    max_seqlen_tensor = None
    if max_seqlen is not None:
        max_seqlen_tensor = _store_val_in_tensor(max_seqlen)

    return torch._nested_view_from_jagged(  # type: ignore[attr-defined]
        values,
        offsets,
        _nt_view_dummy(),
        None,
        ragged_idx,
        min_seqlen_tensor,
        max_seqlen_tensor,
    )  # type: ignore[return-value]