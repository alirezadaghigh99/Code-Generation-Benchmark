def U3(theta, phi, delta):
    r"""
    Arbitrary single qubit unitary.

    .. math::

        U_3(\theta, \phi, \delta) = \begin{bmatrix} \cos(\theta/2) & -\exp(i \delta)\sin(\theta/2) \\
        \exp(i \phi)\sin(\theta/2) & \exp(i (\phi + \delta))\cos(\theta/2) \end{bmatrix}

    Args:dd
        theta (float): polar angle :math:`\theta`
        phi (float): azimuthal angle :math:`\phi`
        delta (float): quantum phase :math:`\delta`
    """
    return math.array(
        [
            [math.cos(theta / 2), -math.exp(delta * 1j) * math.sin(theta / 2)],
            [
                math.exp(phi * 1j) * math.sin(theta / 2),
                math.exp((phi + delta) * 1j) * math.cos(theta / 2),
            ],
        ]
    )

def U2(phi, delta):
    r"""Return the matrix representation of the U2 gate.

    .. math::

        U_2(\phi, \delta) = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -\exp(i \delta)
        \\ \exp(i \phi) & \exp(i (\phi + \delta)) \end{bmatrix}

    Args:dd
        phi (float): azimuthal angle :math:`\phi`
        delta (float): quantum phase :math:`\delta`
    """
    return (
        1
        / math.sqrt(2)
        * math.array(
            [[1.0, -math.exp(delta * 1j)], [math.exp(phi * 1j), math.exp((phi + delta) * 1j)]]
        )
    )

