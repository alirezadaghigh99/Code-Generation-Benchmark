class _CacheData():
    """ Holds cached data that has been retrieved from Tensorflow Event Files and is compressed
    in memory for a single or live training session

    Parameters
    ----------
    labels: list[str]
        The labels for the loss values
    timestamps: :class:`np.ndarray`
        The timestamp of the event step (iteration)
    loss: :class:`np.ndarray`
        The loss values collected for A and B sides for the session
    """
    def __init__(self, labels: list[str], timestamps: np.ndarray, loss: np.ndarray) -> None:
        self.labels = labels
        self._loss = zlib.compress(T.cast(bytes, loss))
        self._timestamps = zlib.compress(T.cast(bytes, timestamps))
        self._timestamps_shape = timestamps.shape
        self._loss_shape = loss.shape

    @property
    def loss(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The loss values for this session """
        retval: np.ndarray = np.frombuffer(zlib.decompress(self._loss), dtype="float32")
        if len(self._loss_shape) > 1:
            retval = retval.reshape(-1, *self._loss_shape[1:])
        return retval

    @property
    def timestamps(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The timestamps for this session """
        retval: np.ndarray = np.frombuffer(zlib.decompress(self._timestamps), dtype="float64")
        if len(self._timestamps_shape) > 1:
            retval = retval.reshape(-1, *self._timestamps_shape[1:])
        return retval

    def add_live_data(self, timestamps: np.ndarray, loss: np.ndarray) -> None:
        """ Add live data to the end of the stored data

        loss: :class:`numpy.ndarray`
            The latest loss values to add to the cache
        timestamps: :class:`numpy.ndarray`
            The latest timestamps  to add to the cache
        """
        new_buffer: list[bytes] = []
        new_shapes: list[tuple[int, ...]] = []
        for data, buffer, dtype, shape in zip([timestamps, loss],
                                              [self._timestamps, self._loss],
                                              ["float64", "float32"],
                                              [self._timestamps_shape, self._loss_shape]):

            old = np.frombuffer(zlib.decompress(buffer), dtype=dtype)
            if data.ndim > 1:
                old = old.reshape(-1, *data.shape[1:])

            new = np.concatenate((old, data))

            logger.debug("old_shape: %s new_shape: %s", shape, new.shape)
            new_buffer.append(zlib.compress(new))
            new_shapes.append(new.shape)
            del old

        self._timestamps = new_buffer[0]
        self._loss = new_buffer[1]
        self._timestamps_shape = new_shapes[0]
        self._loss_shape = new_shapes[1]