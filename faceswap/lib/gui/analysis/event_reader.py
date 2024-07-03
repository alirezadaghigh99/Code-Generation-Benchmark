class _Cache():
    """ Holds parsed Tensorflow log event data in a compressed cache in memory. """
    def __init__(self) -> None:
        logger.debug(parse_class_init(locals()))
        self._data: dict[int, _CacheData] = {}
        self._carry_over: dict[int, EventData] = {}
        self._loss_labels: list[str] = []
        logger.debug("Initialized: %s", self.__class__.__name__)

    def is_cached(self, session_id: int) -> bool:
        """ Check if the given session_id's data is already cached

        Parameters
        ----------
        session_id: int
            The session ID to check

        Returns
        -------
        bool
            ``True`` if the data already exists in the cache otherwise ``False``.
        """
        return self._data.get(session_id) is not None

    def cache_data(self,
                   session_id: int,
                   data: dict[int, EventData],
                   labels: list[str],
                   is_live: bool = False) -> None:
        """ Add a full session's worth of event data to :attr:`_data`.

        Parameters
        ----------
        session_id: int
            The session id to add the data for
        data[int, :class:`EventData`]
            The extracted event data dictionary generated from :class:`_EventParser`
        labels: list[str]
            List of `str` for the labels of each loss value output
        is_live: bool, optional
            ``True`` if the data to be cached is from a live training session otherwise ``False``.
            Default: ``False``
        """
        logger.debug("Caching event data: (session_id: %s, labels: %s, data points: %s, "
                     "is_live: %s)", session_id, labels, len(data), is_live)

        if labels:
            logger.debug("Setting loss labels: %s", labels)
            self._loss_labels = labels

        if not data:
            logger.debug("No data to cache")
            return

        timestamps, loss = self._to_numpy(data, is_live)

        if not is_live or (is_live and not self._data.get(session_id)):
            self._data[session_id] = _CacheData(self._loss_labels, timestamps, loss)
        else:
            self._add_latest_live(session_id, loss, timestamps)

    def _to_numpy(self,
                  data: dict[int, EventData],
                  is_live: bool) -> tuple[np.ndarray, np.ndarray]:
        """ Extract each individual step data into separate numpy arrays for loss and timestamps.

        Timestamps are stored float64 as the extra accuracy is needed for correct timings. Arrays
        are returned at the length of the shortest available data (i.e. truncated records are
        dropped)

        Parameters
        ----------
        data: dict
            The incoming tensorflow event data in dictionary form per step
        is_live: bool, optional
            ``True`` if the data to be cached is from a live training session otherwise ``False``.
            Default: ``False``

        Returns
        -------
        timestamps: :class:`numpy.ndarray`
            float64 array of all iteration's timestamps
        loss: :class:`numpy.ndarray`
            float32 array of all iteration's loss
        """
        if is_live and self._carry_over:
            logger.debug("Processing carry over: %s", self._carry_over)
            self._collect_carry_over(data)

        times, loss = self._process_data(data, is_live)

        if is_live and not all(len(val) == len(self._loss_labels) for val in loss):
            # TODO Many attempts have been made to fix this for live graph logging, and the issue
            # of non-consistent loss record sizes keeps coming up. In the meantime we shall swallow
            # any loss values that are of incorrect length so graph remains functional. This will,
            # most likely, lead to a mismatch on iteration count so a proper fix should be
            # implemented.

            # Timestamps and loss appears to remain consistent with each other, but sometimes loss
            # appears non-consistent. eg (lengths):
            # [2, 2, 2, 2, 2, 2, 2, 0] - last loss collection has zero length
            # [1, 2, 2, 2, 2, 2, 2, 2] - 1st loss collection has 1 length
            # [2, 2, 2, 3, 2, 2, 2] - 4th loss collection has 3 length

            logger.debug("Inconsistent loss found in collection: %s", loss)
            for idx in reversed(range(len(loss))):
                if len(loss[idx]) != len(self._loss_labels):
                    logger.debug("Removing loss/timestamps at position %s", idx)
                    del loss[idx]
                    del times[idx]

        n_times, n_loss = (np.array(times, dtype="float64"), np.array(loss, dtype="float32"))
        logger.debug("Converted to numpy: (data points: %s, timestamps shape: %s, loss shape: %s)",
                     len(data), n_times.shape, n_loss.shape)

        return n_times, n_loss

    def _collect_carry_over(self, data: dict[int, EventData]) -> None:
        """ For live data, collect carried over data from the previous update and merge into the
        current data dictionary.

        Parameters
        ----------
        data: dict[int, :class:`EventData`]
            The latest raw data dictionary
        """
        logger.debug("Carry over keys: %s, data keys: %s", list(self._carry_over), list(data))
        for key in list(self._carry_over):
            if key not in data:
                logger.debug("Carry over found for item %s which does not exist in current "
                             "data: %s. Skipping.", key, list(data))
                continue
            carry_over = self._carry_over.pop(key)
            update = data[key]
            logger.debug("Merging carry over data: %s in to %s", carry_over, update)
            timestamp = update.timestamp
            update.timestamp = carry_over.timestamp if not timestamp else timestamp
            update.loss = carry_over.loss + update.loss
            logger.debug("Merged carry over data: %s", update)

    def _process_data(self,
                      data: dict[int, EventData],
                      is_live: bool) -> tuple[list[float], list[list[float]]]:
        """ Process live update data.

        Live data requires different processing as often we will only have partial data for the
        current step, so we need to cache carried over partial data to be picked up at the next
        query. In addition to this, if training is unexpectedly interrupted, there may also be
        partial data which needs to be cleansed prior to creating a numpy array

        Parameters
        ----------
        data: dict
            The incoming tensorflow event data in dictionary form per step
        is_live: bool
            ``True`` if the data to be cached is from a live training session otherwise ``False``.

        Returns
        -------
        timestamps: tuple
            Cleaned list of complete timestamps for the latest live query
        loss: list
            Cleaned list of complete loss for the latest live query
        """
        timestamps, loss = zip(*[(data[idx].timestamp, data[idx].loss)
                               for idx in sorted(data)])

        l_loss: list[list[float]] = list(loss)
        l_timestamps: list[float] = list(timestamps)

        if len(l_loss[-1]) != len(self._loss_labels):
            logger.debug("Truncated loss found. loss count: %s", len(l_loss))
            idx = sorted(data)[-1]
            if is_live:
                logger.debug("Setting carried over data: %s", data[idx])
                self._carry_over[idx] = data[idx]
            logger.debug("Removing truncated loss: (timestamp: %s, loss: %s)",
                         l_timestamps[-1], loss[-1])
            del l_loss[-1]
            del l_timestamps[-1]

        return l_timestamps, l_loss

    def _add_latest_live(self, session_id: int, loss: np.ndarray, timestamps: np.ndarray) -> None:
        """ Append the latest received live training data to the cached data.

        Parameters
        ----------
        session_id: int
            The training session ID to update the cache for
        loss: :class:`numpy.ndarray`
            The latest loss values returned from the iterator
        timestamps: :class:`numpy.ndarray`
            The latest time stamps returned from the iterator
        """
        logger.debug("Adding live data to cache: (session_id: %s, loss: %s, timestamps: %s)",
                     session_id, loss.shape, timestamps.shape)
        if not np.any(loss) and not np.any(timestamps):
            return

        self._data[session_id].add_live_data(timestamps, loss)

    def get_data(self, session_id: int, metric: T.Literal["loss", "timestamps"]
                 ) -> dict[int, dict[str, np.ndarray | list[str]]] | None:
        """ Retrieve the decompressed cached data from the cache for the given session id.

        Parameters
        ----------
        session_id: int or ``None``
            If session_id is provided, then the cached data for that session is returned. If
            session_id is ``None`` then the cached data for all sessions is returned
        metric: ['loss', 'timestamps']
            The metric to return the data for.

        Returns
        -------
        dict or ``None``
            The `session_id`(s) as key, the values are a dictionary containing the requested
            metric information for each session returned. ``None`` if no data is stored for the
            given session_id
        """
        if session_id is None:
            raw = self._data
        else:
            data = self._data.get(session_id)
            if not data:
                return None
            raw = {session_id: data}

        retval: dict[int, dict[str, np.ndarray | list[str]]] = {}
        for idx, data in raw.items():
            array = data.loss if metric == "loss" else data.timestamps
            val: dict[str, np.ndarray | list[str]] = {str(metric): array}
            if metric == "loss":
                val["labels"] = data.labels
            retval[idx] = val

        logger.debug("Obtained cached data: %s",
                     {session_id: {k: v.shape if isinstance(v, np.ndarray) else v
                                   for k, v in data.items()}
                      for session_id, data in retval.items()})
        return retval