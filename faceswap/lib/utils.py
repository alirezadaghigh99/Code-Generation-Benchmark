def get_backend() -> ValidBackends:
    """ Get the backend that Faceswap is currently configured to use.

    Returns
    -------
    str
        The backend configuration in use by Faceswap. One of  ["cpu", "directml", "nvidia", "rocm",
        "apple_silicon"]

    Example
    -------
    >>> from lib.utils import get_backend
    >>> get_backend()
    'nvidia'
    """
    return _FS_BACKEND

class GetModel():
    """ Check for models in the cache path.

    If available, return the path, if not available, get, unzip and install model

    Parameters
    ----------
    model_filename: str or list
        The name of the model to be loaded (see notes below)
    git_model_id: int
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information

    Notes
    ------
    Models must have a certain naming convention: `<model_name>_v<version_number>.<extension>`
    (eg: `s3fd_v1.pb`).

    Multiple models can exist within the model_filename. They should be passed as a list and follow
    the same naming convention as above. Any differences in filename should occur AFTER the version
    number: `<model_name>_v<version_number><differentiating_information>.<extension>` (eg:
    `["mtcnn_det_v1.1.py", "mtcnn_det_v1.2.py", "mtcnn_det_v1.3.py"]`, `["resnet_ssd_v1.caffemodel"
    ,"resnet_ssd_v1.prototext"]`

    Example
    -------
    >>> from lib.utils import GetModel
    >>> model_downloader = GetModel("s3fd_keras_v2.h5", 11)
    """

    def __init__(self, model_filename: str | list[str], git_model_id: int) -> None:
        self.logger = logging.getLogger(__name__)
        if not isinstance(model_filename, list):
            model_filename = [model_filename]
        self._model_filename = model_filename
        self._cache_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), ".fs_cache")
        self._git_model_id = git_model_id
        self._url_base = "https://github.com/deepfakes-models/faceswap-models/releases/download"
        self._chunk_size = 1024  # Chunk size for downloading and unzipping
        self._retries = 6
        self._get()

    @property
    def _model_full_name(self) -> str:
        """ str: The full model name from the filename(s). """
        common_prefix = os.path.commonprefix(self._model_filename)
        retval = os.path.splitext(common_prefix)[0]
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def _model_name(self) -> str:
        """ str: The model name from the model's full name. """
        retval = self._model_full_name[:self._model_full_name.rfind("_")]
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def _model_version(self) -> int:
        """ int: The model's version number from the model full name. """
        retval = int(self._model_full_name[self._model_full_name.rfind("_") + 2:])
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def model_path(self) -> str | list[str]:
        """ str or list[str]: The model path(s) in the cache folder.

        Example
        -------
        >>> from lib.utils import GetModel
        >>> model_downloader = GetModel("s3fd_keras_v2.h5", 11)
        >>> model_downloader.model_path
        '/path/to/s3fd_keras_v2.h5'
        """
        paths = [os.path.join(self._cache_dir, fname) for fname in self._model_filename]
        retval: str | list[str] = paths[0] if len(paths) == 1 else paths
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def _model_zip_path(self) -> str:
        """ str: The full path to downloaded zip file. """
        retval = os.path.join(self._cache_dir, f"{self._model_full_name}.zip")
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def _model_exists(self) -> bool:
        """ bool: ``True`` if the model exists in the cache folder otherwise ``False``. """
        if isinstance(self.model_path, list):
            retval = all(os.path.exists(pth) for pth in self.model_path)
        else:
            retval = os.path.exists(self.model_path)
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def _url_download(self) -> str:
        """ strL Base download URL for models. """
        tag = f"v{self._git_model_id}.{self._model_version}"
        retval = f"{self._url_base}/{tag}/{self._model_full_name}.zip"
        self.logger.trace("Download url: %s", retval)  # type:ignore[attr-defined]
        return retval

    @property
    def _url_partial_size(self) -> int:
        """ int: How many bytes have already been downloaded. """
        zip_file = self._model_zip_path
        retval = os.path.getsize(zip_file) if os.path.exists(zip_file) else 0
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    def _get(self) -> None:
        """ Check the model exists, if not, download the model, unzip it and place it in the
        model's cache folder. """
        if self._model_exists:
            self.logger.debug("Model exists: %s", self.model_path)
            return
        self._download_model()
        self._unzip_model()
        os.remove(self._model_zip_path)

    def _download_model(self) -> None:
        """ Download the model zip from github to the cache folder. """
        self.logger.info("Downloading model: '%s' from: %s", self._model_name, self._url_download)
        for attempt in range(self._retries):
            try:
                downloaded_size = self._url_partial_size
                req = request.Request(self._url_download)
                if downloaded_size != 0:
                    req.add_header("Range", f"bytes={downloaded_size}-")
                with request.urlopen(req, timeout=10) as response:
                    self.logger.debug("header info: {%s}", response.info())
                    self.logger.debug("Return Code: %s", response.getcode())
                    self._write_zipfile(response, downloaded_size)
                break
            except (socket_error, socket_timeout,
                    urlliberror.HTTPError, urlliberror.URLError) as err:
                if attempt + 1 < self._retries:
                    self.logger.warning("Error downloading model (%s). Retrying %s of %s...",
                                        str(err), attempt + 2, self._retries)
                else:
                    self.logger.error("Failed to download model. Exiting. (Error: '%s', URL: "
                                      "'%s')", str(err), self._url_download)
                    self.logger.info("You can try running again to resume the download.")
                    self.logger.info("Alternatively, you can manually download the model from: %s "
                                     "and unzip the contents to: %s",
                                     self._url_download, self._cache_dir)
                    sys.exit(1)

    def _write_zipfile(self, response: HTTPResponse, downloaded_size: int) -> None:
        """ Write the model zip file to disk.

        Parameters
        ----------
        response: :class:`http.client.HTTPResponse`
            The response from the model download task
        downloaded_size: int
            The amount of bytes downloaded so far
        """
        content_length = response.getheader("content-length")
        content_length = "0" if content_length is None else content_length
        length = int(content_length) + downloaded_size
        if length == downloaded_size:
            self.logger.info("Zip already exists. Skipping download")
            return
        write_type = "wb" if downloaded_size == 0 else "ab"
        with open(self._model_zip_path, write_type) as out_file:
            pbar = tqdm(desc="Downloading",
                        unit="B",
                        total=length,
                        unit_scale=True,
                        unit_divisor=1024)
            if downloaded_size != 0:
                pbar.update(downloaded_size)
            while True:
                buffer = response.read(self._chunk_size)
                if not buffer:
                    break
                pbar.update(len(buffer))
                out_file.write(buffer)
            pbar.close()

    def _unzip_model(self) -> None:
        """ Unzip the model file to the cache folder """
        self.logger.info("Extracting: '%s'", self._model_name)
        try:
            with zipfile.ZipFile(self._model_zip_path, "r") as zip_file:
                self._write_model(zip_file)
        except Exception as err:  # pylint:disable=broad-except
            self.logger.error("Unable to extract model file: %s", str(err))
            sys.exit(1)

    def _write_model(self, zip_file: zipfile.ZipFile) -> None:
        """ Extract files from zip file and write, with progress bar.

        Parameters
        ----------
        zip_file: :class:`zipfile.ZipFile`
            The downloaded model zip file
        """
        length = sum(f.file_size for f in zip_file.infolist())
        fnames = zip_file.namelist()
        self.logger.debug("Zipfile: Filenames: %s, Total Size: %s", fnames, length)
        pbar = tqdm(desc="Decompressing",
                    unit="B",
                    total=length,
                    unit_scale=True,
                    unit_divisor=1024)
        for fname in fnames:
            out_fname = os.path.join(self._cache_dir, fname)
            self.logger.debug("Extracting from: '%s' to '%s'", self._model_zip_path, out_fname)
            zipped = zip_file.open(fname)
            with open(out_fname, "wb") as out_file:
                while True:
                    buffer = zipped.read(self._chunk_size)
                    if not buffer:
                        break
                    pbar.update(len(buffer))
                    out_file.write(buffer)
        pbar.close()

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

