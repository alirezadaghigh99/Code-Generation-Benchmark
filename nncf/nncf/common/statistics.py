class NNCFStatistics(Statistics):
    """
    Groups statistics for all available NNCF compression algorithms.
    Statistics are present only if the algorithm has been started.
    """

    def __init__(self):
        """
        Initializes nncf statistics.
        """
        self._storage = {}

    @property
    def magnitude_sparsity(self) -> Optional[MagnitudeSparsityStatistics]:
        """
        Returns statistics of the magnitude sparsity algorithm. If statistics
        have not been collected, `None` will be returned.

        :return: Instance of the `MagnitudeSparsityStatistics` class.
        """
        return self._storage.get("magnitude_sparsity")

    @property
    def rb_sparsity(self) -> Optional[RBSparsityStatistics]:
        """
        Returns statistics of the RB-sparsity algorithm. If statistics
        have not been collected, `None` will be returned.

        :return: Instance of the `RBSparsityStatistics` class.
        """
        return self._storage.get("rb_sparsity")

    @property
    def movement_sparsity(self) -> Optional[MovementSparsityStatistics]:
        """
        Returns statistics of the movement sparsity algorithm. If statistics
        have not been collected, `None` will be returned.

        :return: Instance of the `MovementSparsityStatistics` class.
        """
        return self._storage.get("movement_sparsity")

    @property
    def const_sparsity(self) -> Optional[ConstSparsityStatistics]:
        """
        Returns statistics of the const sparsity algorithm. If statistics
        have not been collected, `None` will be returned.

        :return: Instance of the `ConstSparsityStatistics` class.
        """
        return self._storage.get("const_sparsity")

    @property
    def quantization(self) -> Optional[QuantizationStatistics]:
        """
        Returns statistics of the quantization algorithm. If statistics
        have not been collected, `None` will be returned.

        :return: Instance of the `QuantizationStatistics` class.
        """
        return self._storage.get("quantization")

    @property
    def filter_pruning(self) -> Optional[FilterPruningStatistics]:
        """
        Returns statistics of the filter pruning algorithm. If statistics
        have not been collected, `None` will be returned.

        :return: Instance of the `FilterPruningStatistics` class.
        """
        return self._storage.get("filter_pruning")

    def register(self, algorithm_name: str, stats: Statistics):
        """
        Registers statistics for the algorithm.

        :param algorithm_name: Name of the algorithm. Should be one of the following
            * magnitude_sparsity
            * rb_sparsity
            * const_sparsity
            * quantization
            * filter_pruning

        :param stats: Statistics of the algorithm.
        """

        available_algorithms = [
            "magnitude_sparsity",
            "rb_sparsity",
            "movement_sparsity",
            "const_sparsity",
            "quantization",
            "filter_pruning",
        ]
        if algorithm_name not in available_algorithms:
            raise ValueError(
                f"Can not register statistics for the algorithm. Unknown name of the algorithm: {algorithm_name}."
            )

        self._storage[algorithm_name] = stats

    def to_str(self) -> str:
        """
        Calls `to_str()` method for all registered statistics of the algorithm and returns
        a sum-up string.

        :return: A representation of the NNCF statistics as a human-readable string.
        """
        pretty_string = "\n\n".join([stats.to_str() for stats in self._storage.values()])
        return pretty_string

    def __iter__(self):
        return iter(self._storage.items())

