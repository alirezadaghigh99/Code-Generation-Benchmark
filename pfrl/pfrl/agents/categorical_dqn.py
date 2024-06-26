def _apply_categorical_projection(y, y_probs, z):
    """Apply categorical projection.

    See Algorithm 1 in https://arxiv.org/abs/1707.06887.

    Args:
        y (ndarray): Values of atoms before projection. Its shape must be
            (batch_size, n_atoms).
        y_probs (ndarray): Probabilities of atoms whose values are y.
            Its shape must be (batch_size, n_atoms).
        z (ndarray): Values of atoms after projection. Its shape must be
            (n_atoms,). It is assumed that the values are sorted in ascending
            order and evenly spaced.

    Returns:
        ndarray: Probabilities of atoms whose values are z.
    """
    batch_size, n_atoms = y.shape
    assert z.shape == (n_atoms,)
    assert y_probs.shape == (batch_size, n_atoms)
    delta_z = z[1] - z[0]
    v_min = z[0]
    v_max = z[-1]
    y = torch.clamp(y, v_min, v_max)

    # bj: (batch_size, n_atoms)
    bj = (y - v_min) / delta_z
    assert bj.shape == (batch_size, n_atoms)
    # Avoid the error caused by inexact delta_z
    bj = torch.clamp(bj, 0, n_atoms - 1)

    # l, u: (batch_size, n_atoms)
    l, u = torch.floor(bj), torch.ceil(bj)
    assert l.shape == (batch_size, n_atoms)
    assert u.shape == (batch_size, n_atoms)

    z_probs = torch.zeros((batch_size, n_atoms), dtype=torch.float32, device=y.device)
    offset = torch.arange(
        0, batch_size * n_atoms, n_atoms, dtype=torch.int32, device=y.device
    )[..., None]
    # Accumulate m_l
    # Note that u - bj in the original paper is replaced with 1 - (bj - l) to
    # deal with the case when bj is an integer, i.e., l = u = bj
    z_probs.view(-1).scatter_add_(
        0, (l.long() + offset).view(-1), (y_probs * (1 - (bj - l))).view(-1)
    )
    # Accumulate m_u
    z_probs.view(-1).scatter_add_(
        0, (u.long() + offset).view(-1), (y_probs * (bj - l)).view(-1)
    )
    return z_probs