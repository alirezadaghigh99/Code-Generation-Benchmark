class Normal(Initializer):
    """
    Initialize parameter sampling from the normal distribution.

    Parameters
    ----------
    mean : int, float
        Mean of the normal distribution.

    std : int, float
        Standard deviation of the normal distribution.

    seed : None or int
        Random seed. Integer value will make results reproducible.
        Defaults to ``None``.

    Methods
    -------
    {Initializer.Methods}
    """
    def __init__(self, mean=0, std=0.01, seed=None):
        self.mean = mean
        self.std = std
        self.seed = seed

    def sample(self, shape, return_array=False):
        if return_array:
            set_numpy_seed(self.seed)
            return np.random.normal(loc=self.mean, scale=self.std, size=shape)

        return tf.random_normal(
            mean=self.mean, stddev=self.std,
            shape=shape, seed=self.seed)

    def __repr__(self):
        return '{}(mean={}, std={})'.format(
            classname(self), self.mean, self.std)

