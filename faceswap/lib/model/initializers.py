class ConvolutionAware(keras.initializers.Initializer):  # type:ignore[name-defined]
    """
    Initializer that generates orthogonal convolution filters in the Fourier space. If this
    initializer is passed a shape that is not 3D or 4D, orthogonal initialization will be used.

    Adapted, fixed and optimized from:
    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/initializers/convaware.py

    Parameters
    ----------
    eps_std: float, optional
        The Standard deviation for the random normal noise used to break symmetry in the inverse
        Fourier transform. Default: 0.05
    seed: int, optional
        Used to seed the random generator. Default: ``None``
    initialized: bool, optional
        This should always be set to ``False``. To avoid Keras re-calculating the values every time
        the model is loaded, this parameter is internally set on first time initialization.
        Default:``False``

    Returns
    -------
    tensor
        The modified kernel weights

    References
    ----------
    Armen Aghajanyan, https://arxiv.org/abs/1702.06295
    """

    def __init__(self, eps_std=0.05, seed=None, initialized=False):
        self.eps_std = eps_std
        self.seed = seed
        self.orthogonal = keras.initializers.Orthogonal()
        self.he_uniform = keras.initializers.he_uniform()
        self.initialized = initialized

    def __call__(self, shape, dtype=None, **kwargs):
        """ Call function for the ICNR initializer.

        Parameters
        ----------
        shape: tuple or list
            The required shape for the output tensor
        dtype: str
            The data type for the tensor

        Returns
        -------
        tensor
            The modified kernel weights
        """
        # TODO Tensorflow appears to pass in a :class:`tensorflow.python.framework.dtypes.DType`
        # object which causes this to error, so currently just reverts to default dtype if a string
        # is not passed in.
        if self.initialized:   # Avoid re-calculating initializer when loading a saved model
            return self.he_uniform(shape, dtype=dtype)
        dtype = K.floatx() if not isinstance(dtype, str) else dtype
        logger.info("Calculating Convolution Aware Initializer for shape: %s", shape)
        rank = len(shape)
        if self.seed is not None:
            np.random.seed(self.seed)

        fan_in, _ = compute_fans(shape)  # pylint:disable=protected-access
        variance = 2 / fan_in

        if rank == 3:
            row, stack_size, filters_size = shape

            transpose_dimensions = (2, 1, 0)
            kernel_shape = (row,)
            correct_ifft = lambda shape, s=[None]: np.fft.irfft(shape, s[0])  # noqa:E501,E731 # pylint:disable=unnecessary-lambda-assignment
            correct_fft = np.fft.rfft

        elif rank == 4:
            row, column, stack_size, filters_size = shape

            transpose_dimensions = (2, 3, 1, 0)
            kernel_shape = (row, column)
            correct_ifft = np.fft.irfft2
            correct_fft = np.fft.rfft2

        elif rank == 5:
            var_x, var_y, var_z, stack_size, filters_size = shape

            transpose_dimensions = (3, 4, 0, 1, 2)
            kernel_shape = (var_x, var_y, var_z)
            correct_fft = np.fft.rfftn
            correct_ifft = np.fft.irfftn

        else:
            self.initialized = True
            return K.variable(self.orthogonal(shape), dtype=dtype)

        kernel_fourier_shape = correct_fft(np.zeros(kernel_shape)).shape

        basis = self._create_basis(filters_size, stack_size, np.prod(kernel_fourier_shape), dtype)
        basis = basis.reshape((filters_size, stack_size,) + kernel_fourier_shape)
        randoms = np.random.normal(0, self.eps_std, basis.shape[:-2] + kernel_shape)
        init = correct_ifft(basis, kernel_shape) + randoms
        init = self._scale_filters(init, variance)
        self.initialized = True
        return K.variable(init.transpose(transpose_dimensions), dtype=dtype, name="conv_aware")

    def _create_basis(self, filters_size, filters, size, dtype):
        """ Create the basis for convolutional aware initialization """
        logger.debug("filters_size: %s, filters: %s, size: %s, dtype: %s",
                     filters_size, filters, size, dtype)
        if size == 1:
            return np.random.normal(0.0, self.eps_std, (filters_size, filters, size))
        nbb = filters // size + 1
        var_a = np.random.normal(0.0, 1.0, (filters_size, nbb, size, size))
        var_a = self._symmetrize(var_a)
        var_u = np.linalg.svd(var_a)[0].transpose(0, 1, 3, 2)
        var_p = np.reshape(var_u, (filters_size, nbb * size, size))[:, :filters, :].astype(dtype)
        return var_p

    @staticmethod
    def _symmetrize(var_a):
        """ Make the given tensor symmetrical. """
        var_b = np.transpose(var_a, axes=(0, 1, 3, 2))
        diag = var_a.diagonal(axis1=2, axis2=3)
        var_c = np.array([[np.diag(arr) for arr in batch] for batch in diag])
        return var_a + var_b - var_c

    @staticmethod
    def _scale_filters(filters, variance):
        """ Scale the given filters. """
        c_var = np.var(filters)
        var_p = np.sqrt(variance / c_var)
        return filters * var_p

    def get_config(self):
        """ Return the Convolutional Aware Initializer configuration.

        Returns
        -------
        dict
            The configuration for ICNR Initialization
        """
        return {"eps_std": self.eps_std,
                "seed": self.seed,
                "initialized": self.initialized}