class ContinuousHellingerDistance(Method):
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

        self._bins: np.ndarray
        self._reference_proba_in_bins: np.ndarray

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Self:
        reference_data = _remove_nans(reference_data)
        len_reference = len(reference_data)

        bins = np.histogram_bin_edges(reference_data, bins='doane')
        reference_proba_in_bins = np.histogram(reference_data, bins=bins)[0] / len_reference
        self._bins = bins
        self._reference_proba_in_bins = reference_proba_in_bins

        return self

    def _calculate(self, data: pd.Series):
        data = _remove_nans(data)
        if data.empty:
            return np.nan
        reference_proba_in_bins = copy(self._reference_proba_in_bins)
        data_proba_in_bins = np.histogram(data, bins=self._bins)[0] / len(data)

        leftover = 1 - np.sum(data_proba_in_bins)
        if leftover > 0:
            data_proba_in_bins = np.append(data_proba_in_bins, leftover)
            reference_proba_in_bins = np.append(reference_proba_in_bins, 0)

        distance = np.sqrt(np.sum((np.sqrt(reference_proba_in_bins) - np.sqrt(data_proba_in_bins)) ** 2)) / np.sqrt(2)

        return distance

class WassersteinDistance(Method):
    """Calculates the Wasserstein Distance between two distributions.

    An alert will be raised for a Chunk if .
    """

    def __init__(self, **kwargs) -> None:
        """Initialize Wasserstein Distance method."""
        super().__init__(
            display_name='Wasserstein distance',
            column_name='wasserstein',
            lower_threshold_limit=0,
            **kwargs,
        )
        """
        Parameters
        ----------
        display_name : str, default='Wasserstein distance'
            The name of the metric. Used to display in plots.
        column_name: str, default='wasserstein'
            The name used to indicate the metric in columns of a DataFrame.
        lower_threshold_limit : float, default=0
            An optional lower threshold for the performance metric.
        """

        self._reference_data: Optional[pd.Series] = None
        self._reference_size: float
        self._bin_width: float
        self._bin_edges: np.ndarray
        self._ref_rel_freqs: Optional[np.ndarray] = None
        self._ref_min: float
        self._ref_max: float
        self._ref_cdf: np.ndarray
        self._fitted = False
        if (
            (not kwargs)
            or ('computation_params' not in kwargs)
            or (self.column_name not in kwargs['computation_params'])
        ):
            self.calculation_method = 'auto'
            self.n_bins = 10_000
        else:
            self.calculation_method = kwargs['computation_params'].get('calculation_method', 'auto')
            self.n_bins = kwargs['computation_params'].get('n_bins', 10_000)

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Self:
        reference_data = _remove_nans(reference_data)
        if (self.calculation_method == 'auto' and len(reference_data) < 10_000) or self.calculation_method == 'exact':
            self._reference_data = reference_data
        else:
            reference_proba_in_bins, self._bin_edges = np.histogram(reference_data, bins=self.n_bins)
            self._ref_rel_freqs = reference_proba_in_bins / len(reference_data)
            self._bin_width = self._bin_edges[1] - self._bin_edges[0]
            self._ref_min = self._bin_edges[0]
            self._ref_max = self._bin_edges[-1]
            self._ref_cdf = np.cumsum(self._ref_rel_freqs)

        self._fitted = True
        self._reference_size = len(reference_data)

        return self

    def _calculate(self, data: pd.Series):
        if not self._fitted:
            raise NotFittedException(
                "tried to call 'calculate' on an unfitted method " f"{self.display_name}. Please run 'fit' first"
            )
        data = _remove_nans(data)
        if data.empty:
            return np.nan
        if (
            self.calculation_method == 'auto' and self._reference_size >= 10_000
        ) or self.calculation_method == 'estimated':
            data_smaller = data[data < self._ref_min]
            data_bigger = data[data > self._ref_max]
            n_smaller = len(data_smaller)
            n_bigger = len(data_bigger)

            if n_smaller > 0:
                amount_smaller = (n_smaller + 1) / len(data)
                smaller_with_first_ref_value = np.concatenate((data_smaller, [self._ref_min]))
                x, y = self._ecdf(smaller_with_first_ref_value)
                term_smaller = np.sum((y)[:-1] * np.diff(x))
                term_smaller = term_smaller * amount_smaller
            else:
                term_smaller, amount_smaller = 0, 0

            if n_bigger > 0:
                amount_bigger = (n_bigger + 1) / len(data)
                bigger_with_last_ref_value = np.concatenate(([self._ref_max], data_bigger))
                x, y = self._ecdf(bigger_with_last_ref_value)
                term_bigger = np.sum((1 - y)[:-1] * np.diff(x))
                term_bigger = term_bigger * amount_bigger
            else:
                term_bigger, amount_bigger = 0, 0

            data_histogram, _ = np.histogram(data, bins=self._bin_edges)
            data_histogram = data_histogram / len(data)

            data_cdf = np.cumsum(data_histogram)
            data_cdf = data_cdf + amount_smaller  # if there's some data on the left-hand side
            term_within = np.sum(np.abs(self._ref_cdf - data_cdf) * self._bin_width)

            distance = term_within + term_smaller + term_bigger
        else:
            distance = wasserstein_distance(self._reference_data, data)

        return distance

    def _ecdf(self, vec: np.ndarray):
        """Custom implementation to calculate ECDF."""
        vec = np.sort(vec)
        x, counts = np.unique(vec, return_counts=True)
        cdf = np.cumsum(counts) / len(vec)
        return x, cdf

