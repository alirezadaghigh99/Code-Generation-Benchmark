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

def beamsplitter(theta, phi):
    r"""Beamsplitter.

    Args:
        theta (float): transmittivity angle (:math:`t=\cos\theta`)
        phi (float): phase angle (:math:`r=e^{i\phi}\sin\theta`)

    Returns:
        array: symplectic transformation matrix
    """
    cp = math.cos(phi)
    sp = math.sin(phi)
    ct = math.cos(theta)
    st = math.sin(theta)

    S = np.array(
        [
            [ct, -cp * st, 0, -st * sp],
            [cp * st, ct, -st * sp, 0],
            [0, st * sp, ct, -cp * st],
            [st * sp, 0, cp * st, ct],
        ]
    )

    return S

