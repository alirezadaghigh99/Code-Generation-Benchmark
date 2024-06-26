def mask_recurrent_state_at(recurrent_state, indices):
    """Return a recurrent state masked at given indices.

    This function can be used to initialize a recurrent state only for a
    certain sequence, not all the sequences.

    Args:
        recurrent_state (object): Batched recurrent state.
        indices (int or array-like of ints): Which recurrent state to mask.

    Returns:
        object: New batched recurrent state.
    """
    if recurrent_state is None:
        return None
    elif isinstance(recurrent_state, torch.Tensor):
        mask = torch.ones_like(recurrent_state)
        mask[:, indices] = 0
        return recurrent_state * mask
    elif isinstance(recurrent_state, tuple):
        return tuple(mask_recurrent_state_at(s, indices) for s in recurrent_state)
    else:
        raise ValueError("Invalid recurrent state: {}".format(recurrent_state))