class _Backend():  # pylint:disable=too-few-public-methods
    """ Return the backend from config/.faceswap of from the `FACESWAP_BACKEND` Environment
    Variable.

    If file doesn't exist and a variable hasn't been set, create the config file. """
    def __init__(self) -> None:
        self._backends: dict[str, ValidBackends] = {"1": "cpu",
                                                    "2": "directml",
                                                    "3": "nvidia",
                                                    "4": "apple_silicon",
                                                    "5": "rocm"}
        self._valid_backends = list(self._backends.values())
        self._config_file = self._get_config_file()
        self.backend = self._get_backend()

    @classmethod
    def _get_config_file(cls) -> str:
        """ Obtain the location of the main Faceswap configuration file.

        Returns
        -------
        str
            The path to the Faceswap configuration file
        """
        pypath = os.path.dirname(os.path.realpath(sys.argv[0]))
        config_file = os.path.join(pypath, "config", ".faceswap")
        return config_file

    def _get_backend(self) -> ValidBackends:
        """ Return the backend from either the `FACESWAP_BACKEND` Environment Variable or from
        the :file:`config/.faceswap` configuration file. If neither of these exist, prompt the user
        to select a backend.

        Returns
        -------
        str
            The backend configuration in use by Faceswap
        """
        # Check if environment variable is set, if so use that
        if "FACESWAP_BACKEND" in os.environ:
            fs_backend = T.cast(ValidBackends, os.environ["FACESWAP_BACKEND"].lower())
            assert fs_backend in T.get_args(ValidBackends), (
                f"Faceswap backend must be one of {T.get_args(ValidBackends)}")
            print(f"Setting Faceswap backend from environment variable to {fs_backend.upper()}")
            return fs_backend
        # Intercept for sphinx docs build
        if sys.argv[0].endswith("sphinx-build"):
            return "nvidia"
        if not os.path.isfile(self._config_file):
            self._configure_backend()
        while True:
            try:
                with open(self._config_file, "r", encoding="utf8") as cnf:
                    config = json.load(cnf)
                break
            except json.decoder.JSONDecodeError:
                self._configure_backend()
                continue
        fs_backend = config.get("backend", "").lower()
        if not fs_backend or fs_backend not in self._backends.values():
            fs_backend = self._configure_backend()
        if current_process().name == "MainProcess":
            print(f"Setting Faceswap backend to {fs_backend.upper()}")
        return fs_backend

    def _configure_backend(self) -> ValidBackends:
        """ Get user input to select the backend that Faceswap should use.

        Returns
        -------
        str
            The backend configuration in use by Faceswap
        """
        print("First time configuration. Please select the required backend")
        while True:
            txt = ", ".join([": ".join([key, val.upper().replace("_", " ")])
                             for key, val in self._backends.items()])
            selection = input(f"{txt}: ")
            if selection not in self._backends:
                print(f"'{selection}' is not a valid selection. Please try again")
                continue
            break
        fs_backend = self._backends[selection]
        config = {"backend": fs_backend}
        with open(self._config_file, "w", encoding="utf8") as cnf:
            json.dump(config, cnf)
        print(f"Faceswap config written to: {self._config_file}")
        return fs_backend