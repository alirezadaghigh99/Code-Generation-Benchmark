class _RepeatingBasisFunction(TransformerMixin, BaseEstimator):
    """Transformer for generating repeating basis functions.

    This transformer generates a set of repeating basis functions for a given input data. Each basis function is
    defined by its center, and the width of the functions is adjusted based on the number of periods. It is
    particularly useful in applications where periodic patterns need to be captured.

    Parameters
    ----------
    n_periods : int, default=12
        The number of repeating periods or basis functions to generate.
    input_range : Tuple[float, float] | List[float] | None, default=None
        The values at which the data repeats itself. For example, for days of the week this is (1,7).
        If `input_range=None` it is inferred from the training data.
    width : float, default=1.0
        The width of the basis functions. This parameter controls how narrow or wide the basis functions are.

    Attributes
    ----------
    bases_ : numpy.ndarray of shape (n_periods,)
        The centers of the repeating basis functions.
    width_ : float
        The adjusted width of the basis functions based on the number of periods and the provided width.
    """

    def __init__(self, n_periods: int = 12, input_range=None, width: float = 1.0):
        self.n_periods = n_periods
        self.input_range = input_range
        self.width = width

    def fit(self, X, y=None):
        """Fit the transformer to the input data and compute the basis functions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data used to compute the basis functions.
        y : array-like of shape (n_samples), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : _RepeatingBasisFunction
            The fitted transformer.
        """
        X = check_array(X, estimator=self)

        # find min and max for standardization if not given explicitly
        if self.input_range is None:
            self.input_range = (X.min(), X.max())

        # exclude the last value because it's identical to the first for repeating basis functions
        self.bases_ = np.linspace(0, 1, self.n_periods + 1)[:-1]

        # curves should narrower (wider) when we have more (fewer) basis functions
        self.width_ = self.width / self.n_periods

        return self

    def transform(self, X):
        """Transform the input data into features based on the repeating basis functions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to be transformed using the basis functions.

        Returns
        -------
        array-like of shape (n_samples, n_periods)
            The transformed data with features generated from the basis functions.

        Raises
        ------
        ValueError
            If X has more than one column, as this transformer only accepts one feature as input.
        """
        X = check_array(X, estimator=self, ensure_2d=True)
        check_is_fitted(self, ["bases_", "width_"])
        # This transformer only accepts one feature as input
        if X.shape[1] != 1:
            raise ValueError(f"X should have exactly one column, it has: {X.shape[1]}")

        # MinMax Scale to 0-1
        X = (X - self.input_range[0]) / (self.input_range[1] - self.input_range[0])

        base_distances = self._array_bases_distances(X, self.bases_)

        # apply rbf function to series for each basis
        return self._rbf(base_distances)

    def _array_base_distance(self, arr: np.ndarray, base: float) -> np.ndarray:
        """Calculate the distances between all array values and the base, where 0 and 1 are assumed to be at the same
        positions

        Parameters
        ----------
        arr : np.ndarray, shape (n_samples,)
            The input array for which distances to the base are calculated.
        base : float
            The base value to which distances are calculated.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            An array of distances between the values in `arr` and the `base`, with consideration of 0 and 1 as
            equivalent positions.
        """
        abs_diff_0 = np.abs(arr - base)
        abs_diff_1 = 1 - abs_diff_0
        concat = np.concatenate((abs_diff_0.reshape(-1, 1), abs_diff_1.reshape(-1, 1)), axis=1)
        final = concat.min(axis=1)
        return final

    def _array_bases_distances(self, array, bases):
        """Calculate distances between all combinations of array and bases values.

        Parameters
        ----------
        array : np.ndarray, shape (n_samples,)
            The input array for which distances to the bases are calculated.
        bases : np.ndarray, shape (n_bases,)
            The bases values to which distances are calculated.

        Returns
        -------
        np.ndarray, shape (n_samples, n_bases)
            An array of distances between the elements of 'array' and the specified 'bases'.
        """
        array = array.reshape(-1, 1)
        bases = bases.reshape(1, -1)

        return np.apply_along_axis(lambda b: self._array_base_distance(array, base=b), axis=0, arr=bases)

    def _rbf(self, arr):
        """Apply the Radial Basis Function (RBF) to the input array.

        Parameters
        ----------
        arr : np.ndarray
            The input array to which the RBF is applied.

        Returns
        -------
        np.ndarray
            An array with the RBF applied to the input array.
        """
        return np.exp(-((arr / self.width_) ** 2))

