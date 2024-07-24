class DFTYamlLoader(DataLoader):
    """
    Creates a `Dataset` object from YAML input files.

    This class provides methods to load and featurize data from a YAML file.
    Although, in this class, we only focus on a specfic input format
    that can be used to perform Density Functional Theory calculations.

    Examples
    --------
    >>> from deepchem.data.data_loader import DFTYamlLoader
    >>> import deepchem as dc
    >>> import pytest
    >>> inputs = 'deepchem/data/tests/dftdata.yaml'
    >>> data = DFTYamlLoader()
    >>> output = data.create_dataset(inputs)

    Notes
    -----
    Format (and example) for the YAML file:

    - e_type : 'ae'
      true_val : '0.09194410469'
      systems : [{
            'moldesc': 'Li 1.5070 0 0; H -1.5070 0 0',
            'basis': '6-311++G(3df,3pd)'
        }]

    Each entry in the YAML file must contain the three parameters : e_type,
    true_val and systems in this particular order.
    One entry object may contain one or more systems.
    This data class does not support/ require an additional featurizer,
    since the datapoints are featurized within the methods.
    To read more about the parameters and their possible values please refer to
    deepchem.feat.dft_data.

    """

    def __init__(self):
        """
        Initialize DFTYAML loader
        """

    def create_dataset(self,
                       inputs: OneOrMany[Any],
                       data_dir: Optional[str] = None,
                       shard_size: Optional[int] = 1) -> Dataset:
        """
        Creates and returns a `Dataset` object by featurizing provided YAML
        files.

        Parameters
        ----------
        input_files: OneOrMany[str]
            List of YAML filenames.
        data_dir: Optional[str], default None
            Name of directory where featurized data is stored.
        shard_size: int, optional (default 1)
            Shard size when loading data.

        Returns
        -------
        DiskDataset
            A `DiskDataset` object containing a featurized representation
            of data from `inputs`.
        """

        def shard_generator():
            entries = self._get_shards(inputs)
            for i, shard in enumerate(entries):
                X = np.array(self._featurize_shard(shard))
                y = X[0].get_true_val()
                w = np.array([X[0].get_weight()])
                ids = np.array([i])
                yield X, y, w, ids

        return DiskDataset.create_dataset(shard_generator(), data_dir)

    def _get_shards(self, inputs):
        """
        Loads and divides the .yaml file into shards.

        Parameters
        ----------
        input_files: str
            .yaml file to be processed.

        Returns
        -------
        data
            list of dictionaries where each dictionary corresponds to one
            shard and is then featurized into one entry object.
        """
        with open(inputs) as f:
            data = yaml.load(f, Loader=SafeLoader)
        return (data)

    def _featurize_shard(self, shard):
        """
        Featurizes shards in the dataset

        Parameters
        ----------
        shard: dict
            Dictionary containing values to initialize the DFTEntry object.

        Returns
        -------
        x: featurized shard (DFTEntry objects)
        """
        try:
            e_type = shard['e_type']
            if 'true_val' in shard.keys():
                true_val = shard['true_val']
            else:
                true_val = '0.0'
            systems = shard['systems']
        except KeyError:
            raise ValueError(
                "Unknown key in yaml file. Please check format for correctness."
            )
        if 'weight' in shard.keys():
            weight = shard['weight']
            x = DFTEntry.create(e_type, true_val, systems, weight)
        else:
            x = DFTEntry.create(e_type, true_val, systems)
        return [x]

class FASTQLoader(DataLoader):
    """Handles loading of FASTQ files.

    FASTQ files are commonly used to hold very large sequence data. It is a variant of FASTA format.
    This class provides convenience files to load FASTQ data and one-hot encode
    the genomic sequences for use in downstream learning tasks.

    Example
    -------
    >>> import os
    >>> from deepchem.feat.molecule_featurizers import OneHotFeaturizer
    >>> from deepchem.data.data_loader import FASTQLoader
    >>> current_dir = os.path.dirname(os.path.abspath(__file__))
    >>> input_file = os.path.join(current_dir, "tests", "sample1.fastq")
    >>> loader = FASTQLoader()
    >>> sequences = loader.create_dataset(input_file)

    See Also
    --------
    `Info on the structure of FASTQ files <https://support.illumina.com/bulletins/2016/04/fastq-files-explained.html>`
    """

    def __init__(self,
                 featurizer: Optional[Featurizer] = None,
                 auto_add_annotations: bool = False,
                 return_quality_scores: bool = False):
        """Initialize FASTQLoader.

        Parameters
        ----------
        featurizer: Featurizer (default: None)
            The Featurizer to be used for the loaded FASTQ data.
            The featurizer is initialized as a OneHotFeaturizer object with charset ("A", "C", "T", "G") and
            max_length = None.
        auto_add_annotations: bool (default False)
            Whether create_dataset will automatically add [CLS] and [SEP] annotations
            to the sequences it reads in order to assist tokenization.
            Keep False if your FASTQ file already includes [CLS] and [SEP] annotations.
        return_quality_scores: bool (default True)
            returns the quality (likelihood) score of the nucleotides in the sequence.
       """

        # Set attributes
        self.auto_add_annotations = auto_add_annotations

        self.user_specified_features = None

        # Handle special featurizer cases
        if isinstance(featurizer,
                      UserDefinedFeaturizer):  # User defined featurizer
            self.user_specified_features = featurizer.feature_fields
        elif featurizer is None:  # Default featurizer
            featurizer = OneHotFeaturizer(charset=["A", "C", "T", "G"],
                                          max_length=None)

        # Set self.featurizer
        self.featurizer = featurizer
        # Set self.return_quality_scores
        self.return_quality_scores = return_quality_scores

    def _get_shards(self,
                    input_files: List[str],
                    shard_size: Optional[int] = 4096) -> Iterator:
        """Defines a generator which returns data for each shard

        Parameters
        ----------
        input_files: List[str]
            List of file names to process
        n_samples:  int, optional
            The number of samples to extract from each variant in the data
        shard_size: int, optional (default 4096)
            The size of a shard of data to process at a time. Here, shard_size is equal to the
            number of variants to fetch. You can think of them as number of rows to get from the
            full dataset.

        Yields
        -------
        Iterator
            Iterator over shards
        """

        return _fastq_load_files(input_files, shard_size)

    def create_dataset(self,
                       input_files: OneOrMany[str],
                       data_dir: Optional[str] = None,
                       shard_size: Optional[int] = 4096) -> DiskDataset:
        """Creates a `Dataset` from input FASTQ files.

        Parameters
        ----------
        input_files: List[str]
            List of fastQ files.
        data_dir: str, optional (default None)
            Name of directory where featurized data is stored.
          shard_size: int, optional (default 4096)

        Returns
        -------
        DiskDataset
            A `DiskDataset` object containing a featurized representation of data
            from `input_files`.
        """
        if isinstance(input_files, str):
            input_files = [input_files]

        def shard_generator():

            for shard_num, shard in enumerate(
                    self._get_shards(input_files, shard_size)):
                if self.return_quality_scores:
                    sequences, quality_scores = _generate_sequences(shard)
                    # Featurize sequences
                    X = self.featurizer(sequences)
                    ids = np.ones(len(X))
                    # (X, y , w, ids)
                    yield X, None, quality_scores, ids
                else:
                    sequences = _generate_sequences(shard)
                    # Featurize sequences
                    X = self.featurizer(sequences)
                    ids = np.ones(len(X))
                    # (X, y , w, ids)
                    yield X, None, None, ids

        def _generate_sequences(shard: List) -> OneOrMany[np.ndarray]:
            """
            Creates a numpy array of annotated FASTQ-format strings.
            """
            assert len(
                shard
            ) % 4 == 0, f'Sharded length not divisible by four: Length of shard = {len(shard)}. File is possibly incomplete'
            sequences: np.ndarray = np.array([], dtype='object')
            if self.return_quality_scores:
                quality_scores: np.ndarray = np.array([], dtype='object')

            # Go through each sequence entity in the fastq_file: each sequence consists of 4 lines
            # First line : header description
            # second line : sequence
            # third line : more description usually the same as the first line
            # fourth line: quality scores of the sequence
            for start_index in range(0, len(shard), 4):
                each_sequence = shard[start_index:start_index + 4]

                # Second line : add sequence to the sequence array
                sequences = _add_sequence(
                    sequences, np.array([each_sequence[1].strip("\n")]))

                # Fourth line
                if self.return_quality_scores:
                    quality_scores = _add_sequence(
                        quality_scores,
                        np.array([each_sequence[3].strip("\n")]))

            if self.return_quality_scores:
                return sequences, quality_scores
            else:
                return sequences

        def _add_sequence(sequences: np.ndarray,
                          sequence: np.ndarray) -> np.ndarray:
            # Handle empty sequence
            if sequence is None or len(sequence) <= 0:
                return np.array([])
            # Annotate start/stop of sequence
            if self.auto_add_annotations:
                sequence = np.insert(sequence, 0, "[CLS]")
                sequence = np.append(sequence, "[SEP]")
            new_sequence = ''.join(sequence)
            new_sequences = np.append(sequences, new_sequence)
            return new_sequences

        return DiskDataset.create_dataset(shard_generator(), data_dir)

