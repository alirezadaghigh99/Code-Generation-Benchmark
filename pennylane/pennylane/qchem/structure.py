def excitations_to_wires(singles, doubles, wires=None):
    r"""Map the indices representing the single and double excitations
    generated with the function :func:`~.excitations` to the wires that
    the Unitary Coupled-Cluster (UCCSD) template will act on.

    Args:
        singles (list[list[int]]): list with the indices ``r``, ``p`` of the two qubits
            representing the single excitation
            :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF}\rangle`
        doubles (list[list[int]]): list with the indices ``s``, ``r``, ``q``, ``p`` of the four
            qubits representing the double excitation
            :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger
            \hat{c}_r \hat{c}_s \vert \mathrm{HF}\rangle`
        wires (Iterable[Any]): Wires of the quantum device. If None, will use consecutive wires.

    The indices :math:`r, s` and :math:`p, q` in these lists correspond, respectively, to the
    occupied and virtual orbitals involved in the generated single and double excitations.

    Returns:
        tuple[list[list[Any]], list[list[list[Any]]]]: lists with the sequence of wires,
        resulting from the single and double excitations, that the Unitary Coupled-Cluster
        (UCCSD) template will act on.

    **Example**

    >>> singles = [[0, 2], [1, 3]]
    >>> doubles = [[0, 1, 2, 3]]
    >>> singles_wires, doubles_wires = excitations_to_wires(singles, doubles)
    >>> print(singles_wires)
    [[0, 1, 2], [1, 2, 3]]
    >>> print(doubles_wires)
    [[[0, 1], [2, 3]]]

    >>> wires=['a0', 'b1', 'c2', 'd3']
    >>> singles_wires, doubles_wires = excitations_to_wires(singles, doubles, wires=wires)
    >>> print(singles_wires)
    [['a0', 'b1', 'c2'], ['b1', 'c2', 'd3']]
    >>> print(doubles_wires)
    [[['a0', 'b1'], ['c2', 'd3']]]
    """

    if (not singles) and (not doubles):
        raise ValueError(
            f"'singles' and 'doubles' lists can not be both empty; "
            f"got singles = {singles}, doubles = {doubles}"
        )

    expected_shape = (2,)
    for single_ in singles:
        if np.array(single_).shape != expected_shape:
            raise ValueError(
                f"Expected entries of 'singles' to be of shape (2,); got {np.array(single_).shape}"
            )

    expected_shape = (4,)
    for double_ in doubles:
        if np.array(double_).shape != expected_shape:
            raise ValueError(
                f"Expected entries of 'doubles' to be of shape (4,); got {np.array(double_).shape}"
            )

    max_idx = 0
    if singles:
        max_idx = np.max(singles)
    if doubles:
        max_idx = max(np.max(doubles), max_idx)

    if wires is None:
        wires = range(max_idx + 1)
    elif len(wires) != max_idx + 1:
        raise ValueError(f"Expected number of wires is {max_idx + 1}; got {len(wires)}")

    singles_wires = []
    for r, p in singles:
        s_wires = [wires[i] for i in range(r, p + 1)]
        singles_wires.append(s_wires)

    doubles_wires = []
    for s, r, q, p in doubles:
        d1_wires = [wires[i] for i in range(s, r + 1)]
        d2_wires = [wires[i] for i in range(q, p + 1)]
        doubles_wires.append([d1_wires, d2_wires])

    return singles_wires, doubles_wiresdef mol_data(identifier, identifier_type="name"):
    r"""Obtain symbols and geometry of a compound from the PubChem Database.

    The `PubChem <https://pubchem.ncbi.nlm.nih.gov>`__ database is one of the largest public
    repositories for information on chemical substances from which symbols and geometry can be
    retrieved for a compound by its name, SMILES, InChI, InChIKey, or PubChem Compound ID (CID) to
    build a molecule object for Hartree-Fock calculations. The retrieved atomic coordinates will be
    converted to `atomic units <https://en.wikipedia.org/wiki/Bohr_radius>`__ for consistency.

    Args:
        identifier (str or int): compound's identifier as required by the PubChem database
        identifier_type (str): type of the provided identifier - name, CAS, CID, SMILES, InChI, InChIKey

    Returns:
        Tuple(list[str], array[float]): symbols and geometry (in Bohr radius) of the compound

    **Example**

    >>> mol_data("BeH2")
    (['Be', 'H', 'H'],
    tensor([[ 4.79404621,  0.29290755,  0.        ],
            [ 3.77945225, -0.29290755,  0.        ],
            [ 5.80882913, -0.29290755,  0.        ]], requires_grad=True))

    >>> mol_data(223, "CID")
    (['N', 'H', 'H', 'H', 'H'],
    tensor([[ 0.        ,  0.        ,  0.        ],
            [ 1.82264085,  0.52836742,  0.40402345],
            [ 0.01417295, -1.67429735, -0.98038991],
            [-0.98927163, -0.22714508,  1.65369933],
            [-0.84773114,  1.373075  , -1.07733286]], requires_grad=True))

    .. details::

        ``mol_data`` can also be used with other chemical identifiers - CAS, SMILES, InChI, InChIKey:

        >>> mol_data("74-82-8", "CAS")
        (['C', 'H', 'H', 'H', 'H'],
        tensor([[ 0.        ,  0.        ,  0.        ],
                [ 1.04709725,  1.51102501,  0.93824902],
                [ 1.29124986, -1.53710323, -0.47923455],
                [-1.47058487, -0.70581271,  1.26460472],
                [-0.86795121,  0.7320799 , -1.7236192 ]], requires_grad=True))

        >>> mol_data("[C]", "SMILES")
        (['C', 'H', 'H', 'H', 'H'],
        tensor([[ 0.        ,  0.        ,  0.        ],
                [ 1.04709725,  1.51102501,  0.93824902],
                [ 1.29124986, -1.53710323, -0.47923455],
                [-1.47058487, -0.70581271,  1.26460472],
                [-0.86795121,  0.7320799 , -1.7236192 ]], requires_grad=True))

        >>> mol_data("InChI=1S/CH4/h1H4", "InChI")
        (['C', 'H', 'H', 'H', 'H'],
        tensor([[ 0.        ,  0.        ,  0.        ],
                [ 1.04709725,  1.51102501,  0.93824902],
                [ 1.29124986, -1.53710323, -0.47923455],
                [-1.47058487, -0.70581271,  1.26460472],
                [-0.86795121,  0.7320799 , -1.7236192 ]], requires_grad=True))

        >>> mol_data("VNWKTOKETHGBQD-UHFFFAOYSA-N", "InChIKey")
        (['C', 'H', 'H', 'H', 'H'],
        tensor([[ 0.        ,  0.        ,  0.        ],
                [ 1.04709725,  1.51102501,  0.93824902],
                [ 1.29124986, -1.53710323, -0.47923455],
                [-1.47058487, -0.70581271,  1.26460472],
                [-0.86795121,  0.7320799 , -1.7236192 ]], requires_grad=True))

    """

    try:
        # pylint: disable=import-outside-toplevel, unused-import, multiple-imports
        import pubchempy as pcp
    except ImportError as Error:
        raise ImportError(
            "This feature requires pubchempy.\nIt can be installed with: pip install pubchempy."
        ) from Error

    # https://gist.github.com/lsauer/1312860/264ae813c2bd2c27a769d261c8c6b38da34e22fb#file-smiles_inchi_annotated-js
    identifier_patterns = {
        "name": re.compile(r"^[a-zA-Z0-9_+-]+$"),
        "cas": re.compile(r"^\d{1,7}\-\d{2}\-\d$"),
        "smiles": re.compile(
            r"^(?!InChI=)(?!\d{1,7}\-\d{2}\-\d)(?![A-Z]{14}\-[A-Z]{10}(\-[A-Z])?)[^J][a-zA-Z0-9@+\-\[\]\(\)\\\/%=#$]{1,}$"
        ),
        "inchi": re.compile(
            r"^InChI\=1S?\/[A-Za-z0-9\.]+(\+[0-9]+)?(\/[cnpqbtmsih][A-Za-z0-9\-\+\(\)\,\/\?\;\.]+)*$"
        ),
        "inchikey": re.compile(r"^[A-Z]{14}\-[A-Z]{10}(\-[A-Z])?"),
    }
    if identifier_type.lower() == "cid":
        cid = int(identifier)
    else:
        if identifier_type.lower() not in identifier_patterns:
            raise ValueError(
                "Specified identifier type is not supported. Supported type are: name, CAS, SMILES, InChI, InChIKey."
            )
        try:
            if identifier_patterns[identifier_type.lower()].search(identifier):
                if identifier_type.lower() == "cas":
                    identifier_type = "name"
                cid = pcp.get_cids(identifier, namespace=identifier_type.lower())[0]
            else:
                raise ValueError(
                    f"Specified identifier doesn't seem to match type: {identifier_type}."
                )
        except (IndexError, pcp.NotFoundError) as exc:
            raise ValueError("Specified molecule does not exist in the PubChem Database.") from exc

    try:
        pcp_molecule = pcp.Compound.from_cid(cid, record_type="3d")
    except pcp.NotFoundError:
        pcp_molecule = pcp.Compound.from_cid(cid, record_type="2d")
    except ValueError as exc:
        raise ValueError("Provided CID (or Identifier) is None.") from exc

    data_mol = pcp_molecule.to_dict(properties=["atoms"])
    symbols = [atom["element"] for atom in data_mol["atoms"]]
    geometry = (
        np.array([[atom["x"], atom["y"], atom.get("z", 0.0)] for atom in data_mol["atoms"]])
        / bohr_angs
    )

    return symbols, geometry