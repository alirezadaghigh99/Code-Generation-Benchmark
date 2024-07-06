def thermal_state(nbar, hbar=2.0):
    r"""Returns a thermal state.

    Args:
        nbar (float): the mean photon number
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`

    Returns:
        array: the thermal state
    """
    means = np.zeros([2])
    state = [(2 * nbar + 1) * np.identity(2) * hbar / 2, means]
    return state

