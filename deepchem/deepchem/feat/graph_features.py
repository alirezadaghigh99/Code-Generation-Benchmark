class ConvMolFeaturizer(MolecularFeaturizer):
    """This class implements the featurization to implement Duvenaud graph convolutions.

    Duvenaud graph convolutions [1]_ construct a vector of descriptors for each
    atom in a molecule. The featurizer computes that vector of local descriptors.

    Examples
    ---------
    >>> import deepchem as dc
    >>> smiles = ["C", "CCC"]
    >>> featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
    >>> f = featurizer.featurize(smiles)
    >>> # Using ConvMolFeaturizer to create featurized fragments derived from molecules of interest.
    ... # This is used only in the context of performing interpretation of models using atomic
    ... # contributions (atom-based model interpretation)
    ... smiles = ["C", "CCC"]
    >>> featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True)
    >>> f = featurizer.featurize(smiles)
    >>> len(f) # contains 2 lists with  featurized fragments from 2 mols
    2

    See Also
    --------
    Detailed examples of `GraphConvModel` interpretation are provided in Tutorial #28

    References
    ---------

    .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for
        learning molecular fingerprints." Advances in neural information
        processing systems. 2015.

    Note
    ----
    This class requires RDKit to be installed.
    """
    name = ['conv_mol']

    def __init__(self,
                 master_atom: bool = False,
                 use_chirality: bool = False,
                 atom_properties: Iterable[str] = [],
                 per_atom_fragmentation: bool = False):
        """
        Parameters
        ----------
        master_atom: Boolean
            if true create a fake atom with bonds to every other atom.
            the initialization is the mean of the other atom features in
            the molecule.  This technique is briefly discussed in
            Neural Message Passing for Quantum Chemistry
            https://arxiv.org/pdf/1704.01212.pdf
        use_chirality: Boolean
            if true then make the resulting atom features aware of the
            chirality of the molecules in question
        atom_properties: list of string or None
            properties in the RDKit Mol object to use as additional
            atom-level features in the larger molecular feature.  If None,
            then no atom-level properties are used.  Properties should be in the
            RDKit mol object should be in the form
            atom XXXXXXXX NAME
            where XXXXXXXX is a zero-padded 8 digit number coresponding to the
            zero-indexed atom index of each atom and NAME is the name of the property
            provided in atom_properties.  So "atom 00000000 sasa" would be the
            name of the molecule level property in mol where the solvent
            accessible surface area of atom 0 would be stored.
        per_atom_fragmentation: Boolean
            If True, then multiple "atom-depleted" versions of each molecule will be created (using featurize() method).
            For each molecule, atoms are removed one at a time and the resulting molecule is featurized.
            The result is a list of ConvMol objects,
            one with each heavy atom removed. This is useful for subsequent model interpretation: finding atoms
            favorable/unfavorable for (modelled) activity. This option is typically used in combination
            with a FlatteningTransformer to split the lists into separate samples.

            Since ConvMol is an object and not a numpy array, need to set dtype to
            object.
        """
        self.dtype = object
        self.master_atom = master_atom
        self.use_chirality = use_chirality
        self.atom_properties = list(atom_properties)
        self.per_atom_fragmentation = per_atom_fragmentation

    def featurize(self,
                  datapoints: Union[RDKitMol, str, Iterable[RDKitMol],
                                    Iterable[str]],
                  log_every_n: int = 1000,
                  **kwargs) -> np.ndarray:
        """
        Override parent: aim is to add handling atom-depleted molecules featurization

        Parameters
        ----------
        datapoints: rdkit.Chem.rdchem.Mol / SMILES string / iterable
            RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
            strings.
        log_every_n: int, default 1000
            Logging messages reported every `log_every_n` samples.

        Returns
        -------
        features: np.ndarray
            A numpy array containing a featurized representation of `datapoints`.
        """
        if 'molecules' in kwargs and datapoints is None:
            datapoints = kwargs.get("molecules")
            raise DeprecationWarning(
                'Molecules is being phased out as a parameter, please pass "datapoints" instead.'
            )

        features = super(ConvMolFeaturizer, self).featurize(datapoints,
                                                            log_every_n=1000)
        if self.per_atom_fragmentation:
            # create temporary valid ids serving to filter out failed featurizations from every sublist
            # of features (i.e. every molecules' frags list), and also totally failed sublists.
            # This makes output digestable by Loaders
            valid_frag_inds = [[
                True if np.array(elt).size > 0 else False for elt in f
            ] for f in features]
            features = np.array(
                [[elt for (is_valid, elt) in zip(l, m) if is_valid
                 ] for (l, m) in zip(valid_frag_inds, features) if any(l)],
                dtype=object)
        return features

    def _get_atom_properties(self, atom):
        """
        For a given input RDKit atom return the values of the properties
        requested when initializing the featurize.  See the __init__ of the
        class for a full description of the names of the properties

        Parameters
        ----------
        atom: RDKit.rdchem.Atom
            Atom to get the properties of
        returns a numpy lists of floats of the same size as self.atom_properties
        """
        values = []
        for prop in self.atom_properties:
            mol_prop_name = str("atom %08d %s" % (atom.GetIdx(), prop))
            try:
                values.append(float(atom.GetOwningMol().GetProp(mol_prop_name)))
            except KeyError:
                raise KeyError("No property %s found in %s in %s" %
                               (mol_prop_name, atom.GetOwningMol(), self))
        return np.array(values)

    def _featurize(self, mol):
        """Encodes mol as a ConvMol object.
        If per_atom_fragmentation is True,
        then for each molecule a list of ConvMolObjects
        will be created"""

        def per_atom(n, a):
            """
            Enumerates fragments resulting from mol object,
            s.t. each fragment = mol with single atom removed (all possible removals are enumerated)
            Goes over nodes, deletes one at a time and updates adjacency list of lists (removes connections to that node)

            Parameters
            ----------
            n: np.array of nodes (number_of_nodes X number_of_features)
            a: list of nested lists of adjacent node pairs

            """
            for i in range(n.shape[0]):
                new_n = np.delete(n, (i), axis=0)
                new_a = []
                for j, node_pair in enumerate(a):
                    if i != j:  # don't need this pair, no more connections to deleted node
                        tmp_node_pair = []
                        for v in node_pair:
                            if v < i:
                                tmp_node_pair.append(v)
                            elif v > i:
                                tmp_node_pair.append(
                                    v - 1
                                )  # renumber node, because of offset after node deletion
                        new_a.append(tmp_node_pair)
                yield new_n, new_a

        # Get the node features
        idx_nodes = [(a.GetIdx(),
                      np.concatenate(
                          (atom_features(a, use_chirality=self.use_chirality),
                           self._get_atom_properties(a))))
                     for a in mol.GetAtoms()]

        idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
        idx, nodes = list(zip(*idx_nodes))

        # Stack nodes into an array
        nodes = np.vstack(nodes)
        if self.master_atom:
            master_atom_features = np.expand_dims(np.mean(nodes, axis=0),
                                                  axis=0)
            nodes = np.concatenate([nodes, master_atom_features], axis=0)

        # Get bond lists with reverse edges included
        edge_list = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
        ]
        # Get canonical adjacency list
        canon_adj_list = [[] for mol_id in range(len(nodes))]
        for edge in edge_list:
            canon_adj_list[edge[0]].append(edge[1])
            canon_adj_list[edge[1]].append(edge[0])

        if self.master_atom:
            fake_atom_index = len(nodes) - 1
            for index in range(len(nodes) - 1):
                canon_adj_list[index].append(fake_atom_index)

        if not self.per_atom_fragmentation:
            return ConvMol(nodes, canon_adj_list)
        else:
            return [ConvMol(n, a) for n, a in per_atom(nodes, canon_adj_list)]

    def feature_length(self):
        return 75 + len(self.atom_properties)

    def __hash__(self):
        atom_properties = tuple(self.atom_properties)
        return hash((self.master_atom, self.use_chirality, atom_properties))

    def __eq__(self, other):
        if not isinstance(self, other.__class__):
            return False
        return self.master_atom == other.master_atom and \
               self.use_chirality == other.use_chirality and \
               tuple(self.atom_properties) == tuple(other.atom_properties)

