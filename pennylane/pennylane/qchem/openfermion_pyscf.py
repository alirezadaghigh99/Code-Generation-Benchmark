def meanfield(
    symbols,
    coordinates,
    name="molecule",
    charge=0,
    mult=1,
    basis="sto-3g",
    package="pyscf",
    outpath=".",
):  # pylint: disable=too-many-arguments
    r"""Generates a file from which the mean field electronic structure
    of the molecule can be retrieved.

    This function uses OpenFermion-PySCF plugins to
    perform the Hartree-Fock (HF) calculation for the polyatomic system using the quantum
    chemistry packages ``PySCF``. The mean field electronic
    structure is saved in an hdf5-formatted file.

    The charge of the molecule can be given to simulate cationic/anionic systems.
    Also, the spin multiplicity can be input to determine the number of unpaired electrons
    occupying the HF orbitals as illustrated in the figure below.

    |

    .. figure:: ../../_static/qchem/hf_references.png
        :align: center
        :width: 50%

    |

    Args:
        symbols (list[str]): symbols of the atomic species in the molecule
        coordinates (array[float]): 1D array with the atomic positions in Cartesian
            coordinates. The coordinates must be given in atomic units and the size of the array
            should be ``3*N`` where ``N`` is the number of atoms.
        name (str): molecule label
        charge (int): net charge of the system
        mult (int): Spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1` for
            :math:`N_\mathrm{unpaired}` unpaired electrons occupying the HF orbitals.
            Possible values for ``mult`` are :math:`1, 2, 3, \ldots`. If not specified,
            a closed-shell HF state is assumed.
        basis (str): Atomic basis set used to represent the HF orbitals. Basis set
            availability per element can be found
            `here <www.psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement>`_
        package (str): Quantum chemistry package used to solve the Hartree-Fock equations.
        outpath (str): path to output directory

    Returns:
        str: absolute path to the file containing the mean field electronic structure

    **Example**

    >>> symbols, coordinates = (['H', 'H'], np.array([0., 0., -0.66140414, 0., 0., 0.66140414]))
    >>> meanfield(symbols, coordinates, name="h2")
    ./h2_pyscf_sto-3g
    """
    openfermion, openfermionpyscf = _import_of()

    if coordinates.size != 3 * len(symbols):
        raise ValueError(
            f"The size of the array 'coordinates' has to be 3*len(symbols) = {3 * len(symbols)};"
            f" got 'coordinates.size' = {coordinates.size}"
        )

    package = package.strip().lower()

    if package not in "pyscf":
        error_message = (
            f"Integration with quantum chemistry package '{package}' is not available. \n Please set"
            f" 'package' to 'pyscf'."
        )
        raise TypeError(error_message)

    filename = name + "_" + package.lower() + "_" + basis.strip()
    path_to_file = os.path.join(outpath.strip(), filename)

    geometry = [
        [symbol, tuple(np.array(coordinates)[3 * i : 3 * i + 3] * bohr_angs)]
        for i, symbol in enumerate(symbols)
    ]

    molecule = openfermion.MolecularData(geometry, basis, mult, charge, filename=path_to_file)

    if package == "pyscf":
        # pylint: disable=import-outside-toplevel
        from openfermionpyscf import run_pyscf

        run_pyscf(molecule, run_scf=1, verbose=0)

    return path_to_file

def two_particle(matrix_elements, core=None, active=None, cutoff=1.0e-12):
    r"""Generates the `FermionOperator <https://github.com/quantumlib/OpenFermion/blob/master/docs/
    tutorials/intro_to_openfermion.ipynb>`_ representing a given two-particle operator
    required to build many-body qubit observables.

    Second quantized two-particle operators are expanded in the basis of single-particle
    states as

    .. math::

        \hat{V} = \frac{1}{2} \sum_{\alpha, \beta, \gamma, \delta}
        \langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle
        ~ &[& \hat{c}_{\alpha\uparrow}^\dagger \hat{c}_{\beta\uparrow}^\dagger
        \hat{c}_{\gamma\uparrow} \hat{c}_{\delta\uparrow} + \hat{c}_{\alpha\uparrow}^\dagger
        \hat{c}_{\beta\downarrow}^\dagger \hat{c}_{\gamma\downarrow} \hat{c}_{\delta\uparrow} \\
        &+& \hat{c}_{\alpha\downarrow}^\dagger \hat{c}_{\beta\uparrow}^\dagger
        \hat{c}_{\gamma\uparrow} \hat{c}_{\delta\downarrow} + \hat{c}_{\alpha\downarrow}^\dagger
        \hat{c}_{\beta\downarrow}^\dagger \hat{c}_{\gamma\downarrow} \hat{c}_{\delta\downarrow}~].

    In the equation above the indices :math:`\alpha, \beta, \gamma, \delta` run over the basis
    of spatial orbitals :math:`\phi_\alpha(r)`. Since the operator :math:`v` acts only on the
    spatial coordinates the spin quantum numbers are indicated explicitly with the up/down arrows.
    The operators :math:`\hat{c}^\dagger` and :math:`\hat{c}` are the particle creation and
    annihilation operators, respectively, and
    :math:`\langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle` denotes the
    matrix elements of the operator :math:`\hat{v}`

    .. math::

        \langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle =
        \int dr_1 \int dr_2 ~ \phi_\alpha^*(r_1) \phi_\beta^*(r_2) ~\hat{v}(r_1, r_2)~
        \phi_\gamma(r_2) \phi_\delta(r_1).

    If an active space is defined (see :func:`~.active_space`), the summation indices
    run over the active orbitals and the contribution due to core orbitals is computed as

    .. math::

        && \hat{V}_\mathrm{core} = v_\mathrm{core} +
        \sum_{\alpha, \beta \in \mathrm{active}} \sum_{i \in \mathrm{core}}
        (2 \langle i, \alpha \vert \hat{v} \vert \beta, i \rangle -
        \langle i, \alpha \vert \hat{v} \vert i, \beta \rangle)~
        [\hat{c}_{\alpha\uparrow}^\dagger \hat{c}_{\beta\uparrow} +
        \hat{c}_{\alpha\downarrow}^\dagger \hat{c}_{\beta\downarrow}] \\
        && v_\mathrm{core} = \sum_{\alpha,\beta \in \mathrm{core}}
        [2 \langle \alpha, \beta \vert \hat{v} \vert \beta, \alpha \rangle -
        \langle \alpha, \beta \vert \hat{v} \vert \alpha, \beta \rangle].

    Args:
        matrix_elements (array[float]): 4D NumPy array with the matrix elements
            :math:`\langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle`
        core (list): indices of core orbitals, i.e., the orbitals that are
            not correlated in the many-body wave function
        active (list): indices of active orbitals, i.e., the orbitals used to
            build the correlated many-body wave function
        cutoff (float): Cutoff value for including matrix elements. The
            matrix elements with absolute value less than ``cutoff`` are neglected.

    Returns:
        FermionOperator: an instance of OpenFermion's ``FermionOperator`` representing the
        two-particle operator :math:`\hat{V}`.

    **Example**

    >>> import numpy as np
    >>> matrix_elements = np.array([[[[ 6.82389533e-01, -1.45716772e-16],
    ...                               [-2.77555756e-17,  1.79000576e-01]],
    ...                              [[-2.77555756e-17,  1.79000576e-16],
    ...                               [ 6.70732778e-01, 0.00000000e+00]]],
    ...                             [[[-1.45716772e-16,  6.70732778e-16],
    ...                               [ 1.79000576e-01, -8.32667268e-17]],
    ...                              [[ 1.79000576e-16, -8.32667268e-17],
    ...                               [ 0.00000000e+00,  7.05105632e-01]]]])
    >>> v_op = two_particle(matrix_elements)
    >>> print(v_op)
    0.3411947665 [0^ 0^ 0 0] +
    0.089500288 [0^ 0^ 2 2] +
    0.3411947665 [0^ 1^ 1 0] +
    0.089500288 [0^ 1^ 3 2] +
    0.335366389 [0^ 2^ 2 0] +
    0.335366389 [0^ 3^ 3 0] +
    0.3411947665 [1^ 0^ 0 1] +
    0.089500288 [1^ 0^ 2 3] +
    0.3411947665 [1^ 1^ 1 1] +
    0.089500288 [1^ 1^ 3 3] +
    0.335366389 [1^ 2^ 2 1] +
    0.335366389 [1^ 3^ 3 1] +
    0.089500288 [2^ 0^ 2 0] +
    0.089500288 [2^ 1^ 3 0] +
    0.352552816 [2^ 2^ 2 2] +
    0.352552816 [2^ 3^ 3 2] +
    0.089500288 [3^ 0^ 2 1] +
    0.089500288 [3^ 1^ 3 1] +
    0.352552816 [3^ 2^ 2 3] +
    0.352552816 [3^ 3^ 3 3]
    """
    openfermion, _ = _import_of()

    orbitals = matrix_elements.shape[0]

    if matrix_elements.ndim != 4:
        raise ValueError(
            f"'matrix_elements' must be a 4D array; got 'matrix_elements.ndim = ' {matrix_elements.ndim}"
        )

    if not core:
        v_core = 0
    else:
        if any(i > orbitals - 1 or i < 0 for i in core):
            raise ValueError(
                f"Indices of core orbitals must be between 0 and {orbitals - 1}; got core = {core}"
            )

        # Compute the contribution of core orbitals
        v_core = sum(
            [
                2 * matrix_elements[alpha, beta, beta, alpha]
                - matrix_elements[alpha, beta, alpha, beta]
                for alpha in core
                for beta in core
            ]
        )

    if active is None:
        if not core:
            active = list(range(orbitals))
        else:
            active = [i for i in range(orbitals) if i not in core]

    if any(i > orbitals - 1 or i < 0 for i in active):
        raise ValueError(
            f"Indices of active orbitals must be between 0 and {orbitals - 1}; got active = {active}"
        )

    # Indices of the matrix elements with absolute values >= cutoff
    indices = np.nonzero(np.abs(matrix_elements) >= cutoff)

    # Single out the indices of active orbitals
    num_indices = len(indices[0])
    quads = [
        [indices[0][i], indices[1][i], indices[2][i], indices[3][i]]
        for i in range(num_indices)
        if all(indices[j][i] in active for j in range(len(indices)))
    ]

    # Build the FermionOperator representing V
    v_op = openfermion.ops.FermionOperator("") * v_core

    # add renormalized (due to core orbitals) "one-particle" operators
    if core:
        for alpha in active:
            for beta in active:
                element = 2 * np.sum(
                    matrix_elements[np.array(core), alpha, beta, np.array(core)]
                ) - np.sum(matrix_elements[np.array(core), alpha, np.array(core), beta])

                # up-up term
                a = 2 * active.index(alpha)
                b = 2 * active.index(beta)
                v_op += openfermion.ops.FermionOperator(((a, 1), (b, 0)), element)

                # down-down term
                v_op += openfermion.ops.FermionOperator(((a + 1, 1), (b + 1, 0)), element)

    # add two-particle operators
    for quad in quads:
        alpha, beta, gamma, delta = quad
        element = matrix_elements[alpha, beta, gamma, delta]

        # up-up-up-up term
        a = 2 * active.index(alpha)
        b = 2 * active.index(beta)
        g = 2 * active.index(gamma)
        d = 2 * active.index(delta)
        v_op += openfermion.ops.FermionOperator(((a, 1), (b, 1), (g, 0), (d, 0)), 0.5 * element)

        # up-down-down-up term
        b = 2 * active.index(beta) + 1
        g = 2 * active.index(gamma) + 1
        v_op += openfermion.ops.FermionOperator(((a, 1), (b, 1), (g, 0), (d, 0)), 0.5 * element)

        # down-up-up-down term
        a = 2 * active.index(alpha) + 1
        b = 2 * active.index(beta)
        g = 2 * active.index(gamma)
        d = 2 * active.index(delta) + 1
        v_op += openfermion.ops.FermionOperator(((a, 1), (b, 1), (g, 0), (d, 0)), 0.5 * element)

        # down-down-down-down term
        b = 2 * active.index(beta) + 1
        g = 2 * active.index(gamma) + 1
        v_op += openfermion.ops.FermionOperator(((a, 1), (b, 1), (g, 0), (d, 0)), 0.5 * element)

    return v_op

def one_particle(matrix_elements, core=None, active=None, cutoff=1.0e-12):
    r"""Generates the `FermionOperator <https://github.com/quantumlib/OpenFermion/blob/master/docs/
    tutorials/intro_to_openfermion.ipynb>`_ representing a given one-particle operator
    required to build many-body qubit observables.

    Second quantized one-particle operators are expanded in the basis of single-particle
    states as

    .. math::

        \hat{T} = \sum_{\alpha, \beta} \langle \alpha \vert \hat{t} \vert \beta \rangle
        [\hat{c}_{\alpha\uparrow}^\dagger \hat{c}_{\beta\uparrow} +
        \hat{c}_{\alpha\downarrow}^\dagger \hat{c}_{\beta\downarrow}].

    In the equation above the indices :math:`\alpha, \beta` run over the basis of spatial
    orbitals :math:`\phi_\alpha(r)`. Since the operator :math:`\hat{t}` acts only on the
    spatial coordinates, the spin quantum numbers are indicated explicitly with the up/down arrows.
    The operators :math:`\hat{c}^\dagger` and :math:`\hat{c}` are the particle creation
    and annihilation operators, respectively, and
    :math:`\langle \alpha \vert \hat{t} \vert \beta \rangle` denotes the matrix elements of
    the operator :math:`\hat{t}`

    .. math::

        \langle \alpha \vert \hat{t} \vert \beta \rangle = \int dr ~ \phi_\alpha^*(r)
        \hat{t}(r) \phi_\beta(r).

    If an active space is defined (see :func:`~.active_space`), the summation indices
    run over the active orbitals and the contribution due to core orbitals is computed as
    :math:`t_\mathrm{core} = 2 \sum_{\alpha\in \mathrm{core}}
    \langle \alpha \vert \hat{t} \vert \alpha \rangle`.

    Args:
        matrix_elements (array[float]): 2D NumPy array with the matrix elements
            :math:`\langle \alpha \vert \hat{t} \vert \beta \rangle`
        core (list): indices of core orbitals, i.e., the orbitals that are
            not correlated in the many-body wave function
        active (list): indices of active orbitals, i.e., the orbitals used to
            build the correlated many-body wave function
        cutoff (float): Cutoff value for including matrix elements. The
            matrix elements with absolute value less than ``cutoff`` are neglected.

    Returns:
        FermionOperator: an instance of OpenFermion's ``FermionOperator`` representing the
        one-particle operator :math:`\hat{T}`.

    **Example**

    >>> import numpy as np
    >>> matrix_elements = np.array([[-1.27785301e+00,  0.00000000e+00],
    ...                             [ 1.52655666e-16, -4.48299696e-01]])
    >>> t_op = one_particle(matrix_elements)
    >>> print(t_op)
    -1.277853006156875 [0^ 0] +
    -1.277853006156875 [1^ 1] +
    -0.44829969610163756 [2^ 2] +
    -0.44829969610163756 [3^ 3]
    """
    openfermion, _ = _import_of()

    orbitals = matrix_elements.shape[0]

    if matrix_elements.ndim != 2:
        raise ValueError(
            f"'matrix_elements' must be a 2D array; got matrix_elements.ndim = {matrix_elements.ndim}"
        )

    if not core:
        t_core = 0
    else:
        if any(i > orbitals - 1 or i < 0 for i in core):
            raise ValueError(
                f"Indices of core orbitals must be between 0 and {orbitals - 1}; got core = {core}"
            )

        # Compute contribution due to core orbitals
        t_core = 2 * np.sum(matrix_elements[np.array(core), np.array(core)])

    if active is None:
        if not core:
            active = list(range(orbitals))
        else:
            active = [i for i in range(orbitals) if i not in core]

    if any(i > orbitals - 1 or i < 0 for i in active):
        raise ValueError(
            f"Indices of active orbitals must be between 0 and {orbitals - 1}; got active = {active}"
        )

    # Indices of the matrix elements with absolute values >= cutoff
    indices = np.nonzero(np.abs(matrix_elements) >= cutoff)

    # Single out the indices of active orbitals
    num_indices = len(indices[0])
    pairs = [
        [indices[0][i], indices[1][i]]
        for i in range(num_indices)
        if all(indices[j][i] in active for j in range(len(indices)))
    ]

    # Build the FermionOperator representing T
    t_op = openfermion.ops.FermionOperator("") * t_core
    for pair in pairs:
        alpha, beta = pair
        element = matrix_elements[alpha, beta]

        # spin-up term
        a = 2 * active.index(alpha)
        b = 2 * active.index(beta)
        t_op += openfermion.ops.FermionOperator(((a, 1), (b, 0)), element)

        # spin-down term
        t_op += openfermion.ops.FermionOperator(((a + 1, 1), (b + 1, 0)), element)

    return t_op

def dipole_of(
    symbols,
    coordinates,
    name="molecule",
    charge=0,
    mult=1,
    basis="sto-3g",
    package="pyscf",
    core=None,
    active=None,
    mapping="jordan_wigner",
    cutoff=1.0e-12,
    outpath=".",
    wires=None,
):
    r"""Computes the electric dipole moment operator in the Pauli basis.

    The second quantized dipole moment operator :math:`\hat{D}` of a molecule is given by

    .. math::

        \hat{D} = -\sum_{\alpha, \beta} \langle \alpha \vert \hat{{\bf r}} \vert \beta \rangle
        [\hat{c}_{\alpha\uparrow}^\dagger \hat{c}_{\beta\uparrow} +
        \hat{c}_{\alpha\downarrow}^\dagger \hat{c}_{\beta\downarrow}] + \hat{D}_\mathrm{n}.

    In the equation above, the indices :math:`\alpha, \beta` run over the basis of Hartree-Fock
    molecular orbitals and the operators :math:`\hat{c}^\dagger` and :math:`\hat{c}` are the
    electron creation and annihilation operators, respectively. The matrix elements of the
    position operator :math:`\hat{{\bf r}}` are computed as

    .. math::

        \langle \alpha \vert \hat{{\bf r}} \vert \beta \rangle = \sum_{i, j}
         C_{\alpha i}^*C_{\beta j} \langle i \vert \hat{{\bf r}} \vert j \rangle,

    where :math:`\vert i \rangle` is the wave function of the atomic orbital,
    :math:`C_{\alpha i}` are the coefficients defining the molecular orbitals,
    and :math:`\langle i \vert \hat{{\bf r}} \vert j \rangle`
    is the representation of operator :math:`\hat{{\bf r}}` in the atomic basis.

    The contribution of the nuclei to the dipole operator is given by

    .. math::

        \hat{D}_\mathrm{n} = \sum_{i=1}^{N_\mathrm{atoms}} Z_i {\bf R}_i \hat{I},


    where :math:`Z_i` and :math:`{\bf R}_i` denote, respectively, the atomic number and the
    nuclear coordinates of the :math:`i`-th atom of the molecule.

    Args:
        symbols (list[str]): symbols of the atomic species in the molecule
        coordinates (array[float]): 1D array with the atomic positions in Cartesian
            coordinates. The coordinates must be given in atomic units and the size of the array
            should be ``3*N`` where ``N`` is the number of atoms.
        name (str): name of the molecule
        charge (int): charge of the molecule
        mult (int): spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1` of the
            Hartree-Fock (HF) state based on the number of unpaired electrons occupying the
            HF orbitals
        basis (str): Atomic basis set used to represent the molecular orbitals. Basis set
            availability per element can be found
            `here <www.psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement>`_
        package (str): quantum chemistry package (pyscf) used to solve the
            mean field electronic structure problem
        core (list): indices of core orbitals
        active (list): indices of active orbitals
        mapping (str): transformation (``'jordan_wigner'``, ``'parity'``, or ``'bravyi_kitaev'``) used to
            map the fermionic operator to the Pauli basis
        cutoff (float): Cutoff value for including the matrix elements
            :math:`\langle \alpha \vert \hat{{\bf r}} \vert \beta \rangle`. The matrix elements
            with absolute value less than ``cutoff`` are neglected.
        outpath (str): path to the directory containing output files
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).

    Returns:
        list[pennylane.Hamiltonian]: the qubit observables corresponding to the components
        :math:`\hat{D}_x`, :math:`\hat{D}_y` and :math:`\hat{D}_z` of the dipole operator in
        atomic units.

    **Example**

    >>> symbols = ["H", "H", "H"]
    >>> coordinates = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0])
    >>> dipole_obs = dipole_of(symbols, coordinates, charge=1)
    >>> print([(h.wires) for h in dipole_obs])
    [<Wires = [0, 1, 2, 3, 4, 5]>, <Wires = [0, 1, 2, 3, 4, 5]>, <Wires = [0]>]

    >>> dipole_obs[0] # x-component of D
    (
        0.4781123173263876 * Z(0)
      + 0.4781123173263876 * Z(1)
      + -0.3913638489489803 * (Y(0) @ Z(1) @ Y(2))
      + -0.3913638489489803 * (X(0) @ Z(1) @ X(2))
      + -0.3913638489489803 * (Y(1) @ Z(2) @ Y(3))
      + -0.3913638489489803 * (X(1) @ Z(2) @ X(3))
      + 0.2661114704527088 * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Y(4))
      + 0.2661114704527088 * (X(0) @ Z(1) @ Z(2) @ Z(3) @ X(4))
      + 0.2661114704527088 * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Y(5))
      + 0.2661114704527088 * (X(1) @ Z(2) @ Z(3) @ Z(4) @ X(5))
      + 0.7144779061810713 * Z(2)
      + 0.7144779061810713 * Z(3)
      + -0.11734958781031017 * (Y(2) @ Z(3) @ Y(4))
      + -0.11734958781031017 * (X(2) @ Z(3) @ X(4))
      + -0.11734958781031017 * (Y(3) @ Z(4) @ Y(5))
      + -0.11734958781031017 * (X(3) @ Z(4) @ X(5))
      + 0.24190977644645698 * Z(4)
      + 0.24190977644645698 * Z(5)
    )
    """
    openfermion, _ = _import_of()

    if mult != 1:
        raise ValueError(
            f"Currently, this functionality is constrained to Hartree-Fock states with spin multiplicity = 1;"
            f" got multiplicity 2S+1 =  {mult}"
        )

    for i in symbols:
        if i not in atomic_numbers:
            raise ValueError(
                f"Currently, only first- or second-row elements of the periodic table are supported;"
                f" got element {i}"
            )

    hf_file = qml.qchem.meanfield(symbols, coordinates, name, charge, mult, basis, package, outpath)

    hf = openfermion.MolecularData(filename=hf_file.strip())

    # Load dipole matrix elements in the atomic basis
    # pylint: disable=import-outside-toplevel
    from pyscf import gto

    mol = gto.M(
        atom=hf.geometry, basis=hf.basis, charge=hf.charge, spin=0.5 * (hf.multiplicity - 1)
    )
    dip_ao = mol.intor_symmetric("int1e_r", comp=3).real

    # Transform dipole matrix elements to the MO basis
    n_orbs = hf.n_orbitals
    c_hf = hf.canonical_orbitals

    dip_mo = np.zeros((3, n_orbs, n_orbs))
    for comp in range(3):
        for alpha in range(n_orbs):
            for beta in range(alpha + 1):
                dip_mo[comp, alpha, beta] = c_hf[:, alpha] @ dip_ao[comp] @ c_hf[:, beta]

        dip_mo[comp] += dip_mo[comp].T - np.diag(np.diag(dip_mo[comp]))

    # Compute the nuclear contribution
    dip_n = np.zeros(3)
    for comp in range(3):
        for i, symb in enumerate(symbols):
            dip_n[comp] += atomic_numbers[symb] * coordinates[3 * i + comp]

    # Build the observable
    dip = []
    for i in range(3):
        fermion_obs = one_particle(dip_mo[i], core=core, active=active, cutoff=cutoff)
        dip.append(observable([-fermion_obs], init_term=dip_n[i], mapping=mapping, wires=wires))

    return dip

