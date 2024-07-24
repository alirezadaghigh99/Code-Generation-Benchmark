class RuntimeLog(AbstractLog):
    """
    This logger accumulates log data in a set of nested dictionaries which are stored in memory. The log data is not automatically saved to the filesystem.

    It can be passed with keyword argument `out` to Monte Carlo drivers in order
    to serialize the output data of the simulation.

    This logger keeps the data in memory, and does not save it to disk. To serialize
    the current content to a file, use the method :py:meth:`~netket.logging.RuntimeLog.serialize`.
    """

    def __init__(self):
        """
        Crates a Runtime Logger.
        """
        self._data = None
        self._old_step = 0

    def __call__(self, step, item, variational_state=None):
        self._data = accum_histories_in_tree(self._data, item, step=step)
        self._old_step = step

    @property
    def data(self):
        """
        The dictionary of logged data.
        """
        return self._data

    def __getitem__(self, key):
        return self.data[key]

    def flush(self, variational_state=None):
        pass

    def serialize(self, path: Union[str, Path, IO]):
        r"""
        Serialize the content of :py:attr:`~netket.logging.RuntimeLog.data` to a file.

        If the file already exists, it is overwritten.

        Args:
            path: The path of the output file. It must be a valid path.
        """
        if isinstance(path, str):
            path = Path(path)

        if isinstance(path, Path):
            parent = path.parent
            filename = path.name
            if not filename.endswith((".log", ".json")):
                filename = filename + ".json"
            path = parent / filename

            with open(path, "wb") as io:
                self._serialize(io)
        else:
            self._serialize(path)

    def _serialize(self, outstream: IO):
        r"""
        Inner method of `serialize`, working on an IO object.
        """
        outstream.write(
            orjson.dumps(
                self.data,
                default=default,
                option=orjson.OPT_SERIALIZE_NUMPY,
            )
        )

    def __repr__(self):
        _str = "RuntimeLog():\n"
        if self.data is not None:
            _str += f" keys = {list(self.data.keys())}"
        return _str

