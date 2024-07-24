class DatabaseService(MultiClientService):
    """Service that controls the underlying MongoDB database."""

    service_name = "db"
    allow_headless = True

    MONGOD_EXE_NAME = "mongod"
    if sys.platform.startswith("win"):
        MONGOD_EXE_NAME += ".exe"

    @property
    def command(self):
        database_dir = focn.load_config().database_dir
        log_path = os.path.join(database_dir, "log", "mongo.log")

        args = [
            DatabaseService.find_mongod(),
            "--dbpath",
            database_dir,
            "--logpath",
            log_path,
            "--port",
            "0",
        ]
        if not sys.platform.startswith("win"):
            args.append("--nounixsocket")

        try:
            etau.ensure_dir(database_dir)
        except:
            raise PermissionError(
                "Database directory `%s` cannot be written to" % database_dir
            )

        try:
            etau.ensure_basedir(log_path)

            if not os.path.isfile(log_path):
                etau.ensure_empty_file(log_path)
        except:
            raise PermissionError(
                "Database log path `%s` cannot be written to" % log_path
            )

        if focx._get_context() == focx._COLAB:
            return ["sudo"] + args

        if focx._get_context() == focx._DATABRICKS:
            return ["sudo"] + args

        return args

    @property
    def port(self):
        return self._wait_for_child_port()

    @staticmethod
    def find_mongod():
        """Returns the path to the `mongod` executable."""
        mongod = os.path.join(
            foc.FIFTYONE_DB_BIN_DIR, DatabaseService.MONGOD_EXE_NAME
        )

        if not os.path.isfile(mongod):
            raise ServiceExecutableNotFound("Could not find `mongod`")

        if not os.access(mongod, os.X_OK):
            raise PermissionError("`mongod` is not executable")

        return mongod

class ServerService(Service):
    """Service that controls the FiftyOne web server."""

    service_name = "server"
    working_dir = foc.SERVER_DIR
    allow_headless = True

    def __init__(self, port, address=None, do_not_track=False):
        self._port = port
        self._address = address
        self._do_not_track = do_not_track
        super().__init__()

    def start(self):
        focx._get_context()  # ensure context is defined
        address = self._address or "127.0.0.1"
        port = self._port

        try:
            server_version = requests.get(
                "http://%s:%i/fiftyone" % (address, port), timeout=2
            ).json()["version"]
        except:
            server_version = None

        if server_version is None:
            # There is likely not a fiftyone server running (remote or local),
            # so start a local server. If there actually is a fiftyone server
            # running that didn't respond to /fiftyone, the local server will
            # fail to start but the app will still connect successfully.
            super().start()
            self._wait_for_child_port(port=port)
        else:
            logger.info(
                "Connected to FiftyOne on port %i at %s.\nIf you are not "
                "connecting to a remote session, you may need to start a new "
                "session and specify a port",
                port,
                address,
            )
            if server_version != foc.VERSION:
                logger.warning(
                    "Server version (%s) does not match client version (%s)",
                    server_version,
                    foc.VERSION,
                )

    @property
    def command(self):
        command = [
            sys.executable,
            "main.py",
            "--port",
            str(self.port),
        ]

        if self.address:
            command += ["--address", self.address]

        return command

    @property
    def port(self):
        return self._port

    @property
    def address(self):
        return self._address

    @property
    def env(self):
        dnt = "1" if self._do_not_track else "0"
        return {"FIFTYONE_DO_NOT_TRACK": dnt}

