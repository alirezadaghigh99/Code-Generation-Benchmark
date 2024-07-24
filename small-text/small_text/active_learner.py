class AbstractPoolBasedActiveLearner(ActiveLearner):

    def query(self, num_samples=10):
        pass

    def update(self, y):
        pass

    @abstractmethod
    def initialize_data(self, indices_initial, y_initial, *args, **kwargs):
        """(Re-)Initializes the current labeled pool.

        This methods needs to be called whenever the underlying data changes, in particularly
        before the first loop.

        Parameters
        ----------
        indices_initial : np.ndarray[int]
            Positional indices pointing at training examples. This is the intially labelled set
            for training an initial classifier.
        y_initial : numpy.ndarray[int] or scipy.sparse.csr_matrix
            The respective labels belonging to the examples referenced by `x_indices_initial`.
        """
        pass

    @property
    @abstractmethod
    def classifier(self):
        pass

    @property
    @abstractmethod
    def query_strategy(self):
        pass

