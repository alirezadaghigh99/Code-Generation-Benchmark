class BalancedSequenceSampler(RecordingSampler):
    """Balanced sampling of sequences of consecutive windows with categorical
    targets.

    Balanced sampling of sequences inspired by the approach of [Perslev2021]_:
    1. Uniformly sample a recording out of the available ones.
    2. Uniformly sample one of the classes.
    3. Sample a window of the corresponding class in the selected recording.
    4. Extract a sequence of windows around the sampled window.

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
        Must contain a column `target` with categorical targets.
    n_windows : int
        Number of consecutive windows in a sequence.
    n_sequences : int
        Number of sequences to sample.
    random_state : np.random.RandomState | int | None
        Random state.

    References
    ----------
    .. [Perslev2021] Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum PJ,
           Igel C. U-Sleep: resilient high-frequency sleep staging. npj Digit.
           Med. 4, 72 (2021).
           https://github.com/perslev/U-Time/blob/master/utime/models/usleep.py
    """

    def __init__(self, metadata, n_windows, n_sequences=10, random_state=None):
        super().__init__(metadata, random_state=random_state)

        self.n_windows = n_windows
        self.n_sequences = n_sequences
        self.info_class = self._init_info(metadata, required_keys=["target"])

    def sample_class(self, rec_ind=None):
        """Return a random class.

        Parameters
        ----------
        rec_ind : int | None
            Index to the recording to sample from. If None, the recording will
            be uniformly sampled across available recordings.

        Returns
        -------
        int
            Sampled class.
        int
            Index to the recording the class was sampled from.
        """
        if rec_ind is None:
            rec_ind = self.sample_recording()
        available_classes = self.info_class.loc[self.info.iloc[rec_ind].name].index
        return self.rng.choice(available_classes), rec_ind

    def _sample_seq_start_ind(self, rec_ind=None, class_ind=None):
        """Sample a sequence and return its start index.

        Sample a window associated with a random recording and a random class
        and randomly sample a sequence with it inside. The function returns the
        index of the beginning of the sequence.

        Parameters
        ----------
        rec_ind : int | None
            Index to the recording to sample from. If None, the recording will
            be uniformly sampled across available recordings.
        class_ind : int | None
            If provided as int, sample a window of the corresponding class. If
            None, the class will be uniformly sampled across available classes.

        Returns
        -------
        int
            Index of the first window of the sequence.
        int
            Corresponding recording index.
        int
            Class of the sampled window.
        """
        if class_ind is None:
            class_ind, rec_ind = self.sample_class(rec_ind)

        rec_inds = self.info.iloc[rec_ind]["index"]
        len_rec_inds = len(rec_inds)

        row = self.info.iloc[rec_ind].name
        if not isinstance(row, tuple):
            # Theres's only one category, e.g. "subject"
            row = tuple([row])
        available_indices = self.info_class.loc[row + tuple([class_ind]), "index"]
        win_ind = self.rng.choice(available_indices)
        win_ind_in_rec = np.where(rec_inds == win_ind)[0][0]

        # Minimum and maximum start indices in the sequence
        min_pos = max(0, win_ind_in_rec - self.n_windows + 1)
        max_pos = min(len_rec_inds - self.n_windows, win_ind_in_rec)
        start_ind = rec_inds[self.rng.randint(min_pos, max_pos + 1)]

        return start_ind, rec_ind, class_ind

    def __len__(self):
        return self.n_sequences

    def __iter__(self):
        for _ in range(self.n_sequences):
            start_ind, _, _ = self._sample_seq_start_ind()
            yield tuple(range(start_ind, start_ind + self.n_windows))

class SequenceSampler(RecordingSampler):
    """Sample sequences of consecutive windows.

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
    n_windows : int
        Number of consecutive windows in a sequence.
    n_windows_stride : int
        Number of windows between two consecutive sequences.
    random : bool
        If True, sample sequences randomly. If False, sample sequences in
        order.
    random_state : np.random.RandomState | int | None
        Random state.

    Attributes
    ----------
    info : pd.DataFrame
        See RecordingSampler.
    file_ids : np.ndarray of ints
        Array of shape (n_sequences,) that indicates from which file each
        sequence comes from. Useful e.g. to do self-ensembling.
    """

    def __init__(
        self, metadata, n_windows, n_windows_stride, randomize=False, random_state=None
    ):
        super().__init__(metadata, random_state=random_state)
        self.randomize = randomize
        self.n_windows = n_windows
        self.n_windows_stride = n_windows_stride
        self.start_inds, self.file_ids = self._compute_seq_start_inds()

    def _compute_seq_start_inds(self):
        """Compute sequence start indices.

        Returns
        -------
        np.ndarray :
            Array of shape (n_sequences,) containing the indices of the first
            windows of possible sequences.
        np.ndarray :
            Array of shape (n_sequences,) containing the unique file number of
            each sequence. Useful e.g. to do self-ensembling.
        """
        end_offset = 1 - self.n_windows if self.n_windows > 1 else None
        start_inds = (
            self.info["index"]
            .apply(lambda x: x[: end_offset : self.n_windows_stride])
            .values
        )
        file_ids = [[i] * len(inds) for i, inds in enumerate(start_inds)]
        return np.concatenate(start_inds), np.concatenate(file_ids)

    def __len__(self):
        return len(self.start_inds)

    def __iter__(self):
        if self.randomize:
            start_inds = self.start_inds.copy()
            self.rng.shuffle(start_inds)
            for start_ind in start_inds:
                yield tuple(range(start_ind, start_ind + self.n_windows))
        else:
            for start_ind in self.start_inds:
                yield tuple(range(start_ind, start_ind + self.n_windows))

