class ProcessFailure:
    """
    Represent the failed process result. When the worker process fails, it may record failure root cause into the file.

    Tries to read the failure timestamp from the provided ``error_file``,
    if the ``error_file`` does not exist, the timestamp is the current
    timestamp (seconds since epoch).

    The ``message`` field is a concise explanation of the failure. If
    the error file exists then the message is obtained from the error file.
    Otherwise one is generated based on the failure signature.

    .. note:: It is assumed that the ``error_file`` is written by
              ``torch.distributed.elastic.multiprocessing.errors.error_handler.ErrorHandler``.
              Otherwise the behavior is undefined.

    """

    local_rank: int
    pid: int
    exitcode: int
    error_file: str
    error_file_data: JSON = field(init=False)
    message: str = field(init=False)
    timestamp: int = field(init=False)

    def __post_init__(self):
        self.error_file_data = _EMPTY_ERROR_DATA
        if os.path.isfile(self.error_file):
            try:
                with open(self.error_file) as fp:
                    self.error_file_data = json.load(fp)
                    logger.debug(
                        "User process failed with error data: %s",
                        json.dumps(self.error_file_data, indent=2),
                    )
                    self.message, self.timestamp = self._get_error_data(
                        self.error_file_data
                    )
            except Exception:
                logger.exception("Failed to parse reply file: %s", self.error_file)
                raise
        else:
            self._set_no_reply_file()

        # make up an informative message if not already present
        if not self.message:
            # signals typically do not generate an error file message
            if self.exitcode < 0:
                self.message = (
                    f"Signal {-self.exitcode} ({self.signal_name()})"
                    f" received by PID {self.pid}"
                )
            else:
                self.message = "To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html"

    def _get_error_data(self, error_file_data: Dict[str, Any]) -> Tuple[str, int]:
        message = error_file_data["message"]
        if isinstance(message, str):
            timestamp = int(error_file_data.get("timestamp", 0))
        else:
            timestamp = int(message["extraInfo"]["timestamp"])
        return (message, timestamp)

    def _set_no_reply_file(self):
        self.error_file = _NOT_AVAILABLE
        self.error_file_data = _EMPTY_ERROR_DATA
        self.message = ""
        self.timestamp = int(time.time())

    def signal_name(self) -> str:
        if self.exitcode < 0:
            # We don't want to kill the parent process trying to find the signal name.
            # if the signal doesn't map to a known name, use not available.
            try:
                return signal.Signals(-self.exitcode).name
            except Exception:
                return _NOT_AVAILABLE
        else:
            return _NOT_AVAILABLE

    def timestamp_isoformat(self):
        """Return timestamp in ISO format (YYYY-MM-DD_HH:MM:SS)."""
        return datetime.fromtimestamp(self.timestamp).isoformat(sep="_")    def signal_name(self) -> str:
        if self.exitcode < 0:
            # We don't want to kill the parent process trying to find the signal name.
            # if the signal doesn't map to a known name, use not available.
            try:
                return signal.Signals(-self.exitcode).name
            except Exception:
                return _NOT_AVAILABLE
        else:
            return _NOT_AVAILABLE