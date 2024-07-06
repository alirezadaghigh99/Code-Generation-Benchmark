def generate_global_features(mol: RDKitMol,
                             features_generators: List[str]) -> np.ndarray:
    """Helper function for generating global features for a RDKit mol based on the given list of feature generators to be used.

    Parameters
    ----------
    mol: RDKitMol
        RDKit molecule to be featurized
    features_generators: List[str]
        List of names of the feature generators to be used featurization

    Returns
    -------
    global_features_array: np.ndarray
        Array of global features

    Examples
    --------
    >>> from rdkit import Chem
    >>> import deepchem as dc
    >>> mol = Chem.MolFromSmiles('C')
    >>> features_generators = ['morgan']
    >>> global_features = dc.feat.molecule_featurizers.dmpnn_featurizer.generate_global_features(mol, features_generators)
    >>> type(global_features)
    <class 'numpy.ndarray'>
    >>> len(global_features)
    2048
    >>> nonzero_features_indices = global_features.nonzero()[0]
    >>> nonzero_features_indices
    array([1264])
    >>> global_features[nonzero_features_indices[0]]
    1.0

    """
    global_features: List[np.ndarray] = []
    available_generators = GraphConvConstants.FEATURE_GENERATORS

    for generator in features_generators:
        if generator in available_generators:
            global_featurizer = available_generators[generator]
            if mol.GetNumHeavyAtoms() > 0:
                global_features.extend(global_featurizer.featurize(mol)[0])
            # for H2
            elif mol.GetNumHeavyAtoms() == 0:
                # not all features are equally long, so used methane as dummy molecule to determine length
                global_features.extend(
                    np.zeros(
                        len(
                            global_featurizer.featurize(
                                Chem.MolFromSmiles('C'))[0])))
        else:
            logger.warning(f"{generator} generator is not available in DMPNN")

    global_features_array: np.ndarray = np.asarray(global_features)

    # Fix nans in features
    replace_token = 0
    global_features_array = np.where(np.isnan(global_features_array),
                                     replace_token, global_features_array)

    return global_features_array

def atom_features(
        atom: RDKitAtom,
        functional_groups: Optional[List[int]] = None,
        only_atom_num: bool = False) -> Sequence[Union[bool, int, float]]:
    """Helper method used to compute atom feature vector.

    Deepchem already contains an atom_features function, however we are defining a new one here due to the need to handle features specific to DMPNN.

    Parameters
    ----------
    atom: RDKitAtom
        Atom to compute features on.
    functional_groups: List[int]
        A k-hot vector indicating the functional groups the atom belongs to.
        Default value is None
    only_atom_num: bool
        Toggle to build a feature vector for an atom containing only the atom number information.

    Returns
    -------
    features: Sequence[Union[bool, int, float]]
        A list of atom features.

    Examples
    --------
    >>> import deepchem as dc
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('C')
    >>> atom = mol.GetAtoms()[0]
    >>> features = dc.feat.molecule_featurizers.dmpnn_featurizer.atom_features(atom)
    >>> type(features)
    <class 'list'>
    >>> len(features)
    133

    """

    if atom is None:
        features: Sequence[Union[bool, int,
                                 float]] = [0] * GraphConvConstants.ATOM_FDIM

    elif only_atom_num:
        features = []
        features += get_atomic_num_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['atomic_num'])
        features += [0] * (
            GraphConvConstants.ATOM_FDIM - GraphConvConstants.MAX_ATOMIC_NUM - 1
        )  # set other features to zero

    else:
        features = []
        features += get_atomic_num_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['atomic_num'])
        features += get_atom_total_degree_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['degree'])
        features += get_atom_formal_charge_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['formal_charge'])
        features += get_atom_chiral_tag_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['chiral_tag'])
        features += get_atom_total_num_Hs_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['num_Hs'])
        features += get_atom_hybridization_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES_HYBRIDIZATION, True)
        features += get_atom_is_in_aromatic_one_hot(atom)
        features = [int(feature) for feature in features]
        features += get_atom_mass(atom)

        if functional_groups is not None:
            features += functional_groups
    return features

def bond_features(bond: RDKitBond) -> Sequence[Union[bool, int, float]]:
    """wrapper function for bond_features() already available in deepchem, used to compute bond feature vector.

    Parameters
    ----------
    bond: RDKitBond
        Bond to compute features on.

    Returns
    -------
    features: Sequence[Union[bool, int, float]]
        A list of bond features.

    Examples
    --------
    >>> import deepchem as dc
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CC')
    >>> bond = mol.GetBondWithIdx(0)
    >>> b_features = dc.feat.molecule_featurizers.dmpnn_featurizer.bond_features(bond)
    >>> type(b_features)
    <class 'list'>
    >>> len(b_features)
    14

    """
    if bond is None:
        b_features: Sequence[Union[
            bool, int, float]] = [1] + [0] * (GraphConvConstants.BOND_FDIM - 1)

    else:
        b_features = [0] + b_Feats(bond, use_extended_chirality=True)
    return b_features

