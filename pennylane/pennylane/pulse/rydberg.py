def rydberg_interaction(
    register: list, wires=None, interaction_coeff: float = 862690, max_distance: float = np.inf
):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the interaction of an ensemble of
    Rydberg atoms due to the Rydberg blockade

    .. math::

        \sum_{i<j} V_{ij} n_i n_j

    where :math:`n_i` corresponds to the projector on the Rydberg state of the atom :math:`i`, and
    :math:`V_{ij}` is the van der Waals potential:

    .. math::

        V_{ij} = \frac{C_6}{R_{ij}^6}

    where :math:`R_{ij}` is the distance between the atoms :math:`i` and :math:`j`, and :math:`C_6`
    is the Rydberg interaction constant, which defaults to :math:`862690 \times 2 \pi \text{MHz } \mu \text{m}^6`.
    The unit of time for the evolution of this Rydberg interaction term is in :math:`\mu \text{s}`.
    This interaction term can be combined with laser drive terms (:func:`~.rydberg_drive`) to create
    a Hamiltonian describing a driven Rydberg atom system.

    .. seealso::

        :func:`~.rydberg_drive`

    Args:
        register (list): list of coordinates of the Rydberg atoms (in micrometers)
        wires (list): List of wires containing the wire values for all the atoms. This list should
            have the same length as ``register``. If ``None``, each atom's wire value will
            correspond to its index in the ``register`` list.
        interaction_coeff (float): Defaults to :math:`862690 \times 2 \pi \text{MHz } \mu\text{m}^6`.
            The value will be multiplied by :math:`2 \pi` internally to convert to angular frequency,
            such that only the value in standard frequency (i.e., 862690 in the default example) should be
            passed.
        max_distance (float): Threshold for distance in :math:`\mu\text{m}` between two Rydberg atoms beyond which their
            contribution to the interaction term is removed from the Hamiltonian.

    Returns:
        :class:`~.ParametrizedHamiltonian`: a :class:`~.ParametrizedHamiltonian` representing the atom interaction

    **Example**

    We create a Hamiltonian describing the van der Waals interaction among 9 Rydberg atoms in a square lattice:

    .. code-block:: python

        atom_coordinates = [[0, 0], [0, 5], [0, 10], [5, 0], [5, 5], [5, 10], [10, 0], [10, 5], [10, 10]]
        wires = [1, 5, 0, 2, 4, 3, 8, 6, 7]
        H_i = qml.pulse.rydberg_interaction(atom_coordinates, wires=wires)

    >>> H_i
    HardwareHamiltonian: terms=36

    As expected, we have :math:`\frac{N(N-1)}{2} = 36` terms for N=9 atoms.

    The interaction term is dependent only on the number and positions of the Rydberg atoms. We can execute this
    pulse program, which corresponds to all driving laser fields being turned off and therefore has no trainable
    parameters. To add a driving laser field, see :func:`~.rydberg_drive`.

    .. code-block:: python

        dev = qml.device("default.qubit.jax", wires=9)

        @qml.qnode(dev, interface="jax")
        def circuit():
            qml.evolve(H_i)([], t=[0, 10])
            return qml.expval(qml.Z(0))

    >>> circuit()
    Array(1., dtype=float32)
    """
    if wires is None:
        wires = list(range(len(register)))
    elif len(wires) != len(register):
        raise ValueError("The length of the wires and the register must match.")

    coeffs = []
    observables = []
    for idx, (pos1, wire1) in enumerate(zip(register[:-1], wires[:-1])):
        for pos2, wire2 in zip(register[(idx + 1) :], wires[(idx + 1) :]):
            atom_distance = np.linalg.norm(qml.math.array(pos2) - pos1)
            if atom_distance > max_distance:
                continue
            # factor 2pi converts interaction coefficient from standard to angular frequency
            Vij = (
                2 * np.pi * interaction_coeff / (abs(atom_distance) ** 6)
            )  # van der Waals potential
            coeffs.append(Vij)
            with qml.QueuingManager.stop_recording():
                observables.append(qml.prod(qml.Projector([1], wire1), qml.Projector([1], wire2)))
                # Rydberg projectors

    settings = RydbergSettings(register, interaction_coeff)

    return HardwareHamiltonian(
        coeffs, observables, settings=settings, reorder_fn=_reorder_parameters
    )