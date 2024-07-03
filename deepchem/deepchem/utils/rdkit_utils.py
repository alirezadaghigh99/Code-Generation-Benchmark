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