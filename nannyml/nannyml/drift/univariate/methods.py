class HellingerDistance(Method):
    """Calculates the Hellinger Distance between two distributions."""

    def __init__(self, **kwargs) -> None:
        """Initialize Hellinger Distance method."""
        super().__init__(
            display_name='Hellinger distance',
            column_name='hellinger',
            lower_threshold_limit=0,
            **kwargs,
        )
        """
        Parameters
        ----------
        display_name : str, default='Hellinger distance'
            The name of the metric. Used to display in plots.
        column_name: str, default='hellinger'
            The name used to indicate the metric in columns of a DataFrame.
        lower_threshold_limit : float, default=0
            An optional lower threshold for the performance metric.
        """

        self._treat_as_type: str
        self._bins: np.ndarray
        self._reference_proba_in_bins: np.ndarray

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Self:
        reference_data = _remove_nans(reference_data)
        if _column_is_categorical(reference_data):
            treat_as_type = 'cat'
        else:
            n_unique_values = len(np.unique(reference_data))
            len_reference = len(reference_data)
            if n_unique_values > 50 or n_unique_values / len_reference > 0.1:
                treat_as_type = 'cont'
            else:
                treat_as_type = 'cat'

        if treat_as_type == 'cont':
            bins = np.histogram_bin_edges(reference_data, bins='doane')
            reference_proba_in_bins = np.histogram(reference_data, bins=bins)[0] / len_reference
            self._bins = bins
            self._reference_proba_in_bins = reference_proba_in_bins
        else:
            reference_unique, reference_counts = np.unique(reference_data, return_counts=True)
            reference_proba_per_unique = reference_counts / len(reference_data)
            self._bins = reference_unique
            self._reference_proba_in_bins = reference_proba_per_unique

        self._treat_as_type = treat_as_type

        return self

    def _calculate(self, data: pd.Series):
        data = _remove_nans(data)
        if data.empty:
            return np.nan
        reference_proba_in_bins = copy(self._reference_proba_in_bins)
        if self._treat_as_type == 'cont':
            len_data = len(data)
            data_proba_in_bins = np.histogram(data, bins=self._bins)[0] / len_data

        else:
            data_unique, data_counts = np.unique(data, return_counts=True)
            data_counts_dic = dict(zip(data_unique, data_counts))
            data_count_on_ref_bins = [data_counts_dic[key] if key in data_counts_dic else 0 for key in self._bins]
            data_proba_in_bins = np.array(data_count_on_ref_bins) / len(data)

        leftover = 1 - np.sum(data_proba_in_bins)
        if leftover > 0:
            data_proba_in_bins = np.append(data_proba_in_bins, leftover)
            reference_proba_in_bins = np.append(reference_proba_in_bins, 0)

        distance = np.sqrt(np.sum((np.sqrt(reference_proba_in_bins) - np.sqrt(data_proba_in_bins)) ** 2)) / np.sqrt(2)

        del reference_proba_in_bins

        return distance