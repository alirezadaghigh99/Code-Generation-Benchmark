def register(cls, key: str) -> Callable:
        def inner_wrapper(wrapped_class: Type[Calibrator]) -> Type[Calibrator]:
            if key in cls._registry:
                warnings.warn(f"re-registering calibrator with key '{key}'")

            cls._registry[key] = wrapped_class
            return wrapped_class

        return inner_wrapper

def create(cls, key: str = 'isotonic', **kwargs):
        """Creates a new Calibrator given a key value and optional keyword args.

        If the provided key equals ``None``, then a new instance of the default Calibrator (IsotonicCalibrator)
        will be returned.

        If a non-existent key is provided an ``InvalidArgumentsException`` is raised.

        Parameters
        ----------
        key : str, default='isotonic'
            The key used to retrieve a Calibrator. When providing a key that is already in the index, the value
            will be overwritten.
        kwargs : dict
            Optional keyword arguments that will be passed along to the function associated with the key.
            It can then use these arguments during the creation of a new Calibrator instance.

        Returns
        -------
        calibrator: Calibrator
            A new instance of a specific Calibrator subclass.

        Examples
        --------
        >>> calibrator = CalibratorFactory.create('isotonic', kwargs={'foo': 'bar'})
        """
        if key not in cls._registry:
            raise InvalidArgumentsException(
                f"calibrator '{key}' unknown. " f"Please provide one of the following: {cls._registry.keys()}"
            )

        calibrator_class = cls._registry.get(key)
        assert calibrator_class

        return calibrator_class(**kwargs)

def _get_bin_index_edges(vector_length: int, bin_count: int) -> List[Tuple[int, int]]:
    """Generates edges of bins for specified vector length and number of bins required.

    Parameters
    ----------
    vector_length : int
        The length of the vector that will be binned using bins.
    bin_count : int
        Number of bins and bin edges that will be generated.

    Returns
    -------
    bin_index_edges : list of tuples with bin edges (indexes)
        See the example below for best intuition.

    Examples
    --------
    >>> get_bin_edge_indexes(20, 4)
    [(0, 5), (5, 10), (10, 15), (15, 20)]

    """
    if vector_length <= 2 * bin_count:
        bin_count = vector_length // 2
        if bin_count < 2:  # pragma: no branch
            raise InvalidArgumentsException(
                "cannot split into minimum of 2 bins. Current sample size "
                f"is {vector_length}, please increase sample size. "
            )

    bin_width = vector_length // bin_count
    bin_edges = np.asarray(range(0, vector_length + 1, bin_width))
    bin_edges[-1] = vector_length
    bin_index_left = bin_edges[:-1]
    bin_index_right = bin_edges[1:]
    bin_index_edges = [(x, y) for x, y in zip(bin_index_left, bin_index_right)]
    return bin_index_edges

def create(cls, key: str = 'isotonic', **kwargs):
        """Creates a new Calibrator given a key value and optional keyword args.

        If the provided key equals ``None``, then a new instance of the default Calibrator (IsotonicCalibrator)
        will be returned.

        If a non-existent key is provided an ``InvalidArgumentsException`` is raised.

        Parameters
        ----------
        key : str, default='isotonic'
            The key used to retrieve a Calibrator. When providing a key that is already in the index, the value
            will be overwritten.
        kwargs : dict
            Optional keyword arguments that will be passed along to the function associated with the key.
            It can then use these arguments during the creation of a new Calibrator instance.

        Returns
        -------
        calibrator: Calibrator
            A new instance of a specific Calibrator subclass.

        Examples
        --------
        >>> calibrator = CalibratorFactory.create('isotonic', kwargs={'foo': 'bar'})
        """
        if key not in cls._registry:
            raise InvalidArgumentsException(
                f"calibrator '{key}' unknown. " f"Please provide one of the following: {cls._registry.keys()}"
            )

        calibrator_class = cls._registry.get(key)
        assert calibrator_class

        return calibrator_class(**kwargs)

