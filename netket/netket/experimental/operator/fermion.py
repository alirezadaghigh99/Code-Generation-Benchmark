def number(
    hilbert: _AbstractHilbert,
    site: int,
    sz: _Optional[int] = None,
    cutoff: float = 1e-10,
    dtype: _Optional[_DType] = None,
):
    """
    Builds the number operator :math:`\\hat{a}^\\dagger\\hat{a}`  acting on the
    `site`-th of the Hilbert space `hilbert`.

    Args:
        hilbert: The hilbert space
        site: the site on which this operator acts
        site: the site on which this operator acts
        sz: spin projection quantum number. This is the eigenvalue of
            the corresponding spin-Z Pauli operator (e.g. `sz = ±1` for
            a spin-1/2, `sz ∈ [-2, -1, 1, 2]` for a spin-3/2 and
            in general `sz ∈ [-2S, -2S + 2, ... 2S-2, 2S]` for
            a spin-S )
        dtype: The datatype to use for the matrix elements.

    Returns:
        The resulting FermionOperator2nd
    """
    idx = _get_index(hilbert, site, sz)
    return _FermionOperator2nd(
        hilbert,
        (
            (
                (idx, 1),
                (idx, 0),
            ),
        ),
        dtype=dtype,
    )

