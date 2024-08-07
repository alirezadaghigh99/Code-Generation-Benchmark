class FileTimerClient(TimerClient):
    """
    Client side of ``FileTimerServer``. This client is meant to be used
    on the same host that the ``FileTimerServer`` is running on and uses
    pid to uniquely identify a worker.
    This client uses a named_pipe to send timer requests to the
    ``FileTimerServer``. This client is a producer while the
    ``FileTimerServer`` is a consumer. Multiple clients can work with
    the same ``FileTimerServer``.

    Args:

        file_path: str, the path of a FIFO special file. ``FileTimerServer``
                        must have created it by calling os.mkfifo().

        signal: signal, the signal to use to kill the process. Using a
                        negative or zero signal will not kill the process.
    """

    def __init__(
        self,
        file_path: str,
        signal=(signal.SIGKILL if sys.platform != "win32" else signal.CTRL_C_EVENT),  # type: ignore[attr-defined]
    ) -> None:
        super().__init__()
        self._file_path = file_path
        self.signal = signal

    def _open_non_blocking(self) -> Optional[io.TextIOWrapper]:
        try:
            fd = os.open(self._file_path, os.O_WRONLY | os.O_NONBLOCK)
            return os.fdopen(fd, "wt")
        except Exception:
            return None

    def _send_request(self, request: FileTimerRequest) -> None:
        # The server may have crashed or may haven't started yet.
        # In such case, calling open() in blocking model blocks the client.
        # To avoid such issue, open it in non-blocking mode, and an OSError will
        # be raised if the server is not there.
        file = self._open_non_blocking()
        if file is None:
            raise BrokenPipeError(
                "Could not send the FileTimerRequest because FileTimerServer is not available."
            )
        with file:
            json_request = request.to_json()
            # Write request with no greater than select.PIPE_BUF is guarantee to be atomic.
            if len(json_request) > select.PIPE_BUF:
                raise RuntimeError(
                    f"FileTimerRequest larger than {select.PIPE_BUF} bytes "
                    f"is not supported: {json_request}"
                )
            file.write(json_request + "\n")

    def acquire(self, scope_id: str, expiration_time: float) -> None:
        self._send_request(
            request=FileTimerRequest(
                worker_pid=os.getpid(),
                scope_id=scope_id,
                expiration_time=expiration_time,
                signal=self.signal,
            ),
        )

    def release(self, scope_id: str) -> None:
        self._send_request(
            request=FileTimerRequest(
                worker_pid=os.getpid(), scope_id=scope_id, expiration_time=-1, signal=0
            ),
        )

class FileTimerRequest(TimerRequest):
    """
    Data object representing a countdown timer acquisition and release
    that is used between the ``FileTimerClient`` and ``FileTimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.
    ``signal`` is the signal to reap the worker process from the server
    process.
    """

    __slots__ = ["version", "worker_pid", "scope_id", "expiration_time", "signal"]

    def __init__(
        self, worker_pid: int, scope_id: str, expiration_time: float, signal: int = 0
    ) -> None:
        self.version = 1
        self.worker_pid = worker_pid
        self.scope_id = scope_id
        self.expiration_time = expiration_time
        self.signal = signal

    def __eq__(self, other) -> bool:
        if isinstance(other, FileTimerRequest):
            return (
                self.version == other.version
                and self.worker_pid == other.worker_pid
                and self.scope_id == other.scope_id
                and self.expiration_time == other.expiration_time
                and self.signal == other.signal
            )
        return False

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "pid": self.worker_pid,
                "scope_id": self.scope_id,
                "expiration_time": self.expiration_time,
                "signal": self.signal,
            },
        )

