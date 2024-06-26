class _State():  # pylint:disable=too-few-public-methods
    """ Parses the state file in the current model directory, if the model is training, and
    formats the content into a human readable format. """
    def __init__(self) -> None:
        self._model_dir = self._get_arg("-m", "--model-dir")
        self._trainer = self._get_arg("-t", "--trainer")
        self.state_file = self._get_state_file()

    @property
    def _is_training(self) -> bool:
        """ bool: ``True`` if this function has been called during a training session
        otherwise ``False``. """
        return len(sys.argv) > 1 and sys.argv[1].lower() == "train"

    @staticmethod
    def _get_arg(*args: str) -> str | None:
        """ Obtain the value for a given command line option from sys.argv.

        Returns
        -------
        str or ``None``
            The value of the given command line option, if it exists, otherwise ``None``
        """
        cmd = sys.argv
        for opt in args:
            if opt in cmd:
                return cmd[cmd.index(opt) + 1]
        return None

    def _get_state_file(self) -> str:
        """ Parses the model's state file and compiles the contents into a human readable string.

        Returns
        -------
        str
            The state file formatted into a human readable format
        """
        if not self._is_training or self._model_dir is None or self._trainer is None:
            return ""
        fname = os.path.join(self._model_dir, f"{self._trainer}_state.json")
        if not os.path.isfile(fname):
            return ""

        retval = "\n\n=============== State File =================\n"
        with open(fname, "r", encoding="utf-8", errors="replace") as sfile:
            retval += sfile.read()
        return retval