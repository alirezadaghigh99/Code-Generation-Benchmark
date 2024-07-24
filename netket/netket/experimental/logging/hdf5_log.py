class HDF5Log(AbstractLog):
    r"""
    HDF5 Logger, that can be passed with keyword argument `logger` to Monte
    Carlo drivers in order to serialize the output data of the simulation.

    The logger has support for scalar numbers, NumPy/JAX arrays, and netket.stats.Stats objects.
    These are stored as individual groups within a HDF5 file, under the main group `data/`:

    - scalars are stored as a group with one dataset values of shape :code:`(n_steps,)` containing the logged values,
    - arrays are stored in the same way, but with values having shape :code:`(n_steps, *array_shape)`,
    - netket.stats.Stats are stored as a group containing each field :code:`(Mean, Variance, etc...)` as a separate dataset.

    Importantly, each group has a dataset :code:`iters`, which tracks the
    iteration number of the logged quantity.

    If the model state is serialized, then it is serialized as a dataset in the group `variational_state/`.
    The target of the serialization is the parameters PyTree of the variational state (stored in the group
    `variational_state/parameters`), and the rest of the variational state variables (stored in the group
    `variational_state/model_state`)

    Data can be deserialized by calling :code:`f = h5py.File(filename, 'r')` and
    inspecting the datasets as a dictionary, i.e. :code:`f['data/energy/Mean']`

    .. note::
        The API of this logger is covered by our Semantic Versioning API guarantees. However, the structure of the
        logged files is not, and might change in the future while we iterate on this logger. If you think that we
        could improve the output format of this logger, please open an issue on the NetKet repository and let us
        know.

    """

    def __init__(
        self,
        path: str,
        mode: str = "write",
        save_params: bool = True,
        save_params_every: int = 1,
    ):
        """
        Construct a HDF5 Logger.

        Args:
            path: the name of the output files before the extension
            mode: Specify the behaviour in case the file already exists at this
                path. Options are
                - `[w]rite`: (default) overwrites file if it already exists;
                - `[x]` or `fail`: fails if file already exists;
            save_params: bool flag indicating whether variables of the variational state
                should be serialized at some interval
            save_params_every: every how many iterations should machine parameters be
                flushed to file
        """
        import h5py  # noqa: F401

        super().__init__()

        if not ((mode == "write") or (mode == "append") or (mode == "fail")):
            raise ValueError(
                "Mode not recognized: should be one of `[w]rite`, `[a]ppend` or"
                "`[x]`(fail)."
            )
        mode = _mode_shorthands[mode]

        if not path.endswith((".h5", ".hdf5")):
            path = path + ".h5"

        if os.path.exists(path) and mode == "x":
            raise ValueError(
                "Output file already exists. Either delete it manually or"
                "change `path`."
            )

        dir_name = os.path.dirname(path)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)

        self._file_mode = mode
        self._file_name = path
        self._writer = None

        self._save_params = save_params
        self._save_params_every = save_params_every
        self._steps_notsaved_params = 0

    def _init_output_file(self):
        import h5py

        self._writer = h5py.File(self._file_name, self._file_mode)

    def __call__(self, step, log_data, variational_state):
        if self._writer is None:
            self._init_output_file()

        tree_log(log_data, "data", self._writer, iter=step)

        if self._steps_notsaved_params % self._save_params_every == 0:
            variables = variational_state.variables
            # TODO: remove - FrozenDict are deprecated
            if isinstance(variables, FrozenDict):
                variables = variables.unfreeze()

            _, params = fpop(variables, "params")
            binary_data = to_bytes(variables)
            tree = {"model_state": binary_data, "parameters": params, "iter": step}
            tree_log(tree, "variational_state", self._writer)
            self._steps_notsaved_params = 0

        self._writer.flush()
        self._steps_notsaved_params += 1

    def flush(self, variational_state=None):
        """
        Writes to file the content of this logger.

        Args:
            variational_state: optionally also writes the parameters of the machine.
        """
        if self._writer is not None:
            self._writer.flush()

    def __del__(self):
        if hasattr(self, "_writer"):
            self.flush()

    def __repr__(self):
        _str = f"HDF5Log('{self._file_name}', mode={self._file_mode}"
        return _str

