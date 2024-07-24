def from_str(cls, vm: str) -> Union["Std", Dict[int, "Std"]]:
        """
        Example:
        ::

         from_str("0") -> Std.NONE
         from_str("1") -> Std.OUT
         from_str("0:3,1:0,2:1,3:2") -> {0: Std.ALL, 1: Std.NONE, 2: Std.OUT, 3: Std.ERR}

        Any other input raises an exception
        """

        def to_std(v: str) -> Std:  # type: ignore[return]
            s = Std(int(v))
            if s in Std:
                return s
            # return None -> should NEVER reach here since we regex check input

        if re.match(_VALUE_REGEX, vm):  # vm is a number (e.g. 0)
            return to_std(vm)
        elif re.match(_MAPPING_REGEX, vm):  # vm is a mapping (e.g. 0:1,1:2)
            d: Dict[int, Std] = {}
            for m in vm.split(","):
                i, v = m.split(":")
                d[int(i)] = to_std(v)
            return d
        else:
            raise ValueError(
                f"{vm} does not match: <{_VALUE_REGEX}> or <{_MAPPING_REGEX}>"
            )

class SignalException(Exception):
    """
    Exception is raised inside the torchelastic agent process by the termination handler
    if the death signal got received by the process.
    """

    def __init__(self, msg: str, sigval: signal.Signals) -> None:
        super().__init__(msg)
        self.sigval = sigval

class DefaultLogsSpecs(LogsSpecs):
    """
    Default LogsSpecs implementation:

    - `log_dir` will be created if it doesn't exist
    - Generates nested folders for each attempt and rank.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        redirects: Union[Std, Dict[int, Std]] = Std.NONE,
        tee: Union[Std, Dict[int, Std]] = Std.NONE,
        local_ranks_filter: Optional[Set[int]] = None,
    ) -> None:
        if log_dir != os.devnull:
            if not log_dir:
                log_dir = tempfile.mkdtemp(prefix="torchelastic_")
            elif not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            else:
                if os.path.isfile(log_dir):
                    raise NotADirectoryError(f"log_dir: {log_dir} is a file")
        super().__init__(log_dir, redirects, tee, local_ranks_filter)
        # initialized only once
        self._run_log_dir = None

    @property
    def root_log_dir(self) -> str:
        return str(self._root_log_dir)

    def _make_log_dir(self, log_dir: Optional[str], rdzv_run_id: str):
        base_log_dir = log_dir or tempfile.mkdtemp(prefix="torchelastic_")
        os.makedirs(base_log_dir, exist_ok=True)
        dir = tempfile.mkdtemp(prefix=f"{rdzv_run_id}_", dir=base_log_dir)
        logger.info("log directory set to: %s", dir)
        return dir

    def reify(
        self,
        envs: Dict[int, Dict[str, str]],
    ) -> LogsDest:
        """
        Uses following scheme to build log destination paths:

        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/stdout.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/stderr.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/error.json`
        """
        nprocs = len(envs)
        global_env = {}  # use only to query properies that are not dependent on a rank
        if nprocs > 0:
            global_env = envs[0]
        else:
            logger.warning(
                "Empty envs map provided when defining logging destinations."
            )
        # Keys are always defined, but values can be missing in unit tests
        run_id = global_env.get("TORCHELASTIC_RUN_ID", "test_run_id")
        restart_count = global_env.get("TORCHELASTIC_RESTART_COUNT", "0")

        attempt_log_dir: str = ""
        if self._root_log_dir != os.devnull:
            if not self._run_log_dir:
                self._run_log_dir = self._make_log_dir(self._root_log_dir, run_id)

            attempt_log_dir = os.path.join(self._run_log_dir, f"attempt_{restart_count}")  # type: ignore[call-overload]
            shutil.rmtree(attempt_log_dir, ignore_errors=True)
            os.makedirs(attempt_log_dir)

        if self._root_log_dir == os.devnull:
            attempt_log_dir = os.devnull

        # create subdirs for each local rank in the logs_dir
        # logs_dir
        #       |- 0
        #          |- error.json
        #          |- stdout.log
        #          |- stderr.log
        #       |- ...
        #       |- (nprocs-1)
        redirs = to_map(self._redirects, nprocs)
        ts = to_map(self._tee, nprocs)

        # to tee stdout/stderr we first redirect into a file
        # then tail -f stdout.log/stderr.log so add tee settings to redirects
        for local_rank, tee_std in ts.items():
            redirect_std = redirs[local_rank]
            redirs[local_rank] = redirect_std | tee_std

        SYS_STREAM = ""  # special case to indicate to output to console
        stdouts = dict.fromkeys(range(nprocs), SYS_STREAM)
        stderrs = dict.fromkeys(range(nprocs), SYS_STREAM)
        tee_stdouts: Dict[int, str] = {}
        tee_stderrs: Dict[int, str] = {}
        error_files = {}

        for local_rank in range(nprocs):
            if attempt_log_dir == os.devnull:
                tee_stdouts[local_rank] = os.devnull
                tee_stderrs[local_rank] = os.devnull
                error_files[local_rank] = os.devnull
                envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = ""
            else:
                clogdir = os.path.join(attempt_log_dir, str(local_rank))
                os.mkdir(clogdir)

                rd = redirs[local_rank]
                if (rd & Std.OUT) == Std.OUT:
                    stdouts[local_rank] = os.path.join(clogdir, "stdout.log")
                if (rd & Std.ERR) == Std.ERR:
                    stderrs[local_rank] = os.path.join(clogdir, "stderr.log")

                t = ts[local_rank]
                if t & Std.OUT == Std.OUT:
                    tee_stdouts[local_rank] = stdouts[local_rank]
                if t & Std.ERR == Std.ERR:
                    tee_stderrs[local_rank] = stderrs[local_rank]

                if (
                    self._local_ranks_filter
                    and local_rank not in self._local_ranks_filter
                ):
                    # If stream is tee'd, only write to file, but don't tail
                    if local_rank in tee_stdouts:
                        tee_stdouts.pop(local_rank, None)
                    if local_rank in tee_stderrs:
                        tee_stderrs.pop(local_rank, None)

                    # If stream is not redirected, don't print
                    if stdouts[local_rank] == SYS_STREAM:
                        stdouts[local_rank] = os.devnull
                    if stderrs[local_rank] == SYS_STREAM:
                        stderrs[local_rank] = os.devnull

                error_file = os.path.join(clogdir, "error.json")
                error_files[local_rank] = error_file
                logger.info(
                    "Setting worker%s reply file to: %s", local_rank, error_file
                )
                envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = error_file

        return LogsDest(stdouts, stderrs, tee_stdouts, tee_stderrs, error_files)

    def __repr__(self) -> str:
        return (
            f"DefaultLogsSpecs(root_log_dir={self._root_log_dir}, redirects={self._redirects}, "
            f"tee={self._tee}, local_ranks_filter={self._local_ranks_filter})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DefaultLogsSpecs):
            return False

        return (
            self._root_log_dir == other._root_log_dir
            and self._redirects == other._redirects
            and self._tee == other._tee
            and self._local_ranks_filter == other._local_ranks_filter
        )

class DefaultLogsSpecs(LogsSpecs):
    """
    Default LogsSpecs implementation:

    - `log_dir` will be created if it doesn't exist
    - Generates nested folders for each attempt and rank.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        redirects: Union[Std, Dict[int, Std]] = Std.NONE,
        tee: Union[Std, Dict[int, Std]] = Std.NONE,
        local_ranks_filter: Optional[Set[int]] = None,
    ) -> None:
        if log_dir != os.devnull:
            if not log_dir:
                log_dir = tempfile.mkdtemp(prefix="torchelastic_")
            elif not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            else:
                if os.path.isfile(log_dir):
                    raise NotADirectoryError(f"log_dir: {log_dir} is a file")
        super().__init__(log_dir, redirects, tee, local_ranks_filter)
        # initialized only once
        self._run_log_dir = None

    @property
    def root_log_dir(self) -> str:
        return str(self._root_log_dir)

    def _make_log_dir(self, log_dir: Optional[str], rdzv_run_id: str):
        base_log_dir = log_dir or tempfile.mkdtemp(prefix="torchelastic_")
        os.makedirs(base_log_dir, exist_ok=True)
        dir = tempfile.mkdtemp(prefix=f"{rdzv_run_id}_", dir=base_log_dir)
        logger.info("log directory set to: %s", dir)
        return dir

    def reify(
        self,
        envs: Dict[int, Dict[str, str]],
    ) -> LogsDest:
        """
        Uses following scheme to build log destination paths:

        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/stdout.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/stderr.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/error.json`
        """
        nprocs = len(envs)
        global_env = {}  # use only to query properies that are not dependent on a rank
        if nprocs > 0:
            global_env = envs[0]
        else:
            logger.warning(
                "Empty envs map provided when defining logging destinations."
            )
        # Keys are always defined, but values can be missing in unit tests
        run_id = global_env.get("TORCHELASTIC_RUN_ID", "test_run_id")
        restart_count = global_env.get("TORCHELASTIC_RESTART_COUNT", "0")

        attempt_log_dir: str = ""
        if self._root_log_dir != os.devnull:
            if not self._run_log_dir:
                self._run_log_dir = self._make_log_dir(self._root_log_dir, run_id)

            attempt_log_dir = os.path.join(self._run_log_dir, f"attempt_{restart_count}")  # type: ignore[call-overload]
            shutil.rmtree(attempt_log_dir, ignore_errors=True)
            os.makedirs(attempt_log_dir)

        if self._root_log_dir == os.devnull:
            attempt_log_dir = os.devnull

        # create subdirs for each local rank in the logs_dir
        # logs_dir
        #       |- 0
        #          |- error.json
        #          |- stdout.log
        #          |- stderr.log
        #       |- ...
        #       |- (nprocs-1)
        redirs = to_map(self._redirects, nprocs)
        ts = to_map(self._tee, nprocs)

        # to tee stdout/stderr we first redirect into a file
        # then tail -f stdout.log/stderr.log so add tee settings to redirects
        for local_rank, tee_std in ts.items():
            redirect_std = redirs[local_rank]
            redirs[local_rank] = redirect_std | tee_std

        SYS_STREAM = ""  # special case to indicate to output to console
        stdouts = dict.fromkeys(range(nprocs), SYS_STREAM)
        stderrs = dict.fromkeys(range(nprocs), SYS_STREAM)
        tee_stdouts: Dict[int, str] = {}
        tee_stderrs: Dict[int, str] = {}
        error_files = {}

        for local_rank in range(nprocs):
            if attempt_log_dir == os.devnull:
                tee_stdouts[local_rank] = os.devnull
                tee_stderrs[local_rank] = os.devnull
                error_files[local_rank] = os.devnull
                envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = ""
            else:
                clogdir = os.path.join(attempt_log_dir, str(local_rank))
                os.mkdir(clogdir)

                rd = redirs[local_rank]
                if (rd & Std.OUT) == Std.OUT:
                    stdouts[local_rank] = os.path.join(clogdir, "stdout.log")
                if (rd & Std.ERR) == Std.ERR:
                    stderrs[local_rank] = os.path.join(clogdir, "stderr.log")

                t = ts[local_rank]
                if t & Std.OUT == Std.OUT:
                    tee_stdouts[local_rank] = stdouts[local_rank]
                if t & Std.ERR == Std.ERR:
                    tee_stderrs[local_rank] = stderrs[local_rank]

                if (
                    self._local_ranks_filter
                    and local_rank not in self._local_ranks_filter
                ):
                    # If stream is tee'd, only write to file, but don't tail
                    if local_rank in tee_stdouts:
                        tee_stdouts.pop(local_rank, None)
                    if local_rank in tee_stderrs:
                        tee_stderrs.pop(local_rank, None)

                    # If stream is not redirected, don't print
                    if stdouts[local_rank] == SYS_STREAM:
                        stdouts[local_rank] = os.devnull
                    if stderrs[local_rank] == SYS_STREAM:
                        stderrs[local_rank] = os.devnull

                error_file = os.path.join(clogdir, "error.json")
                error_files[local_rank] = error_file
                logger.info(
                    "Setting worker%s reply file to: %s", local_rank, error_file
                )
                envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = error_file

        return LogsDest(stdouts, stderrs, tee_stdouts, tee_stderrs, error_files)

    def __repr__(self) -> str:
        return (
            f"DefaultLogsSpecs(root_log_dir={self._root_log_dir}, redirects={self._redirects}, "
            f"tee={self._tee}, local_ranks_filter={self._local_ranks_filter})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DefaultLogsSpecs):
            return False

        return (
            self._root_log_dir == other._root_log_dir
            and self._redirects == other._redirects
            and self._tee == other._tee
            and self._local_ranks_filter == other._local_ranks_filter
        )

class SignalException(Exception):
    """
    Exception is raised inside the torchelastic agent process by the termination handler
    if the death signal got received by the process.
    """

    def __init__(self, msg: str, sigval: signal.Signals) -> None:
        super().__init__(msg)
        self.sigval = sigval

