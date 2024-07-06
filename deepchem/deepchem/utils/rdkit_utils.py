def load_molecule(molecule_file,
                  add_hydrogens=True,
                  calc_charges=True,
                  sanitize=True,
                  is_protein=False):
    """Converts molecule file to (xyz-coords, obmol object)

    Given molecule_file, returns a tuple of xyz coords of molecule
    and an rdkit object representing that molecule in that order `(xyz,
    rdkit_mol)`. This ordering convention is used in the code in a few
    places.

    Parameters
    ----------
    molecule_file: str
        filename for molecule
    add_hydrogens: bool, optional (default True)
        If True, add hydrogens via pdbfixer
    calc_charges: bool, optional (default True)
        If True, add charges via rdkit
    sanitize: bool, optional (default False)
        If True, sanitize molecules via rdkit
    is_protein: bool, optional (default False)
        If True`, this molecule is loaded as a protein. This flag will
        affect some of the cleanup procedures applied.

    Returns
    -------
    Tuple (xyz, mol) if file contains single molecule. Else returns a
    list of the tuples for the separate molecules in this list.

    Note
    ----
    This function requires RDKit to be installed.
    """
    from rdkit import Chem
    from_pdb = False
    if ".mol2" in molecule_file:
        my_mol = Chem.MolFromMol2File(molecule_file,
                                      sanitize=False,
                                      removeHs=False)
    elif ".sdf" in molecule_file:
        suppl = Chem.SDMolSupplier(str(molecule_file), sanitize=False)
        # TODO: This is wrong. Should return all molecules
        my_mol = suppl[0]
    elif ".pdbqt" in molecule_file:
        pdb_block = pdbqt_to_pdb(molecule_file)
        my_mol = Chem.MolFromPDBBlock(str(pdb_block),
                                      sanitize=False,
                                      removeHs=False)
        from_pdb = True
    elif ".pdb" in molecule_file:
        my_mol = Chem.MolFromPDBFile(str(molecule_file),
                                     sanitize=False,
                                     removeHs=False)
        from_pdb = True  # noqa: F841
    else:
        raise ValueError("Unrecognized file type for %s" % str(molecule_file))

    if my_mol is None:
        raise ValueError("Unable to read non None Molecule Object")

    if add_hydrogens or calc_charges:
        my_mol = apply_pdbfixer(my_mol,
                                hydrogenate=add_hydrogens,
                                is_protein=is_protein)
    if sanitize:
        try:
            Chem.SanitizeMol(my_mol)
        # TODO: Ideally we should catch AtomValenceException but Travis seems to choke on it for some reason.
        except:
            logger.warning("Mol %s failed sanitization" %
                           Chem.MolToSmiles(my_mol))
    if calc_charges:
        # This updates in place
        compute_charges(my_mol)

    xyz = get_xyz_from_mol(my_mol)

    return xyz, my_mol

def load_complex(molecular_complex: OneOrMany[str],
                 add_hydrogens: bool = True,
                 calc_charges: bool = True,
                 sanitize: bool = True) -> List[Tuple[np.ndarray, RDKitMol]]:
    """Loads a molecular complex.

    Given some representation of a molecular complex, returns a list of
    tuples, where each tuple contains (xyz coords, rdkit object) for
    that constituent molecule in the complex.

    For now, assumes that molecular_complex is a tuple of filenames.

    Parameters
    ----------
    molecular_complex: list or str
        If list, each entry should be a filename for a constituent
        molecule in complex. If str, should be the filename of a file that
        holds the full complex.
    add_hydrogens: bool, optional
        If true, add hydrogens via pdbfixer
    calc_charges: bool, optional
        If true, add charges via rdkit
    sanitize: bool, optional
        If true, sanitize molecules via rdkit

    Returns
    -------
    List of tuples (xyz, mol)

    Note
    ----
    This function requires RDKit to be installed.
    """
    if isinstance(molecular_complex, str):
        molecular_complex = [molecular_complex]
    fragments: List = []
    for mol in molecular_complex:
        loaded = load_molecule(mol,
                               add_hydrogens=add_hydrogens,
                               calc_charges=calc_charges,
                               sanitize=sanitize)
        if isinstance(loaded, list):
            fragments += loaded
        else:
            fragments.append(loaded)
    return fragments

