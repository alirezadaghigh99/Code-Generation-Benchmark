def construct_hydrogen_bonding_info(mol: RDKitMol) -> List[Tuple[int, str]]:
    """Construct hydrogen bonding infos about a molecule.

    Parameters
    ---------
    mol: rdkit.Chem.rdchem.Mol
        RDKit mol object

    Returns
    -------
    List[Tuple[int, str]]
        A list of tuple `(atom_index, hydrogen_bonding_type)`.
        The `hydrogen_bonding_type` value is "Acceptor" or "Donor".
    """
    factory = _ChemicalFeaturesFactory.get_instance()
    feats = factory.GetFeaturesForMol(mol)
    hydrogen_bonding = []
    for f in feats:
        hydrogen_bonding.append((f.GetAtomIds()[0], f.GetFamily()))
    return hydrogen_bonding

