def _get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())

class RunResult:
    """Return results of the worker executions.

    Run results follow an "all-or-nothing" policy where the run is successful if and
    only if ALL local workers managed by this agent complete successfully.

    If the result is successful (e.g. ``is_failed() = False``) then the ``return_values``
    field contains the outputs (return values) of the workers managed by THIS agent mapped
    by their GLOBAL ranks. That is ``result.return_values[0]`` is the return value of
    global rank 0.

    .. note:: ``return_values`` are only meaningful for when the worker entrypoint
              is a function. Workers specified as a binary entrypoint do not canonically
              have a return value and the ``return_values`` field is meaningless and
              may be empty.

    If ``is_failed()`` returns ``True`` then the ``failures`` field contains the
    failure information, again, mapped by the GLOBAL rank of the worker that failed.

    The keys in ``return_values`` and ``failures`` are mutually exclusive, that is,
    a worker's final state can only be one of: succeeded, failed. Workers intentionally
    terminated by the agent according to the agent's restart policy, are not represented
    in either ``return_values`` nor ``failures``.
    """

    state: WorkerState
    return_values: Dict[int, Any] = field(default_factory=dict)
    failures: Dict[int, ProcessFailure] = field(default_factory=dict)

    def is_failed(self) -> bool:
        return self.state == WorkerState.FAILED

class _RoleInstanceInfo:
    """The class is used by the agent to exchange the information with other agents.

    The information is used to determine the rank of the workers that agent
    manages in heterogeneous environments, where different agents can have
    different number of workers.
    """

    __slots__ = ["role", "rank", "local_world_size"]

    def __init__(self, role: str, rank: int, local_world_size: int):
        r"""Initialize the agent class instance.

        Args:
            role (str): user-defined role for the workers with this spec
            rank (int): the rank of the agent
            local_world_size (int): number of local workers to run
        """
        self.role = role
        self.rank = rank
        self.local_world_size = local_world_size

    def serialize(self) -> bytes:
        dict_data = {
            "role": self.role,
            "rank": self.rank,
            "local_world_size": self.local_world_size,
        }
        return json.dumps(dict_data).encode(encoding="UTF-8")

    @staticmethod
    def deserialize(data: bytes):
        dict_data = json.loads(data.decode(encoding="UTF-8"))
        return _RoleInstanceInfo(
            dict_data["role"], dict_data["rank"], dict_data["local_world_size"]
        )

    @staticmethod
    def compare(obj1, obj2) -> int:
        if obj1.role == obj2.role:
            return obj1.rank - obj2.rank
        elif obj1.role > obj2.role:
            return 1
        else:
            return -1

    @staticmethod
    def find_role_boundaries(roles_infos: List, role: str) -> Tuple[int, int]:
        start_idx, end_idx = -1, -1
        for idx, role_info in enumerate(roles_infos):
            if role_info.role == role:
                if start_idx == -1:
                    start_idx = idx
                end_idx = idx
        return (start_idx, end_idx)

class WorkerSpec:
    """Blueprint information about a particular type of worker.

    For a given role, there must only exist a single worker spec.
    Worker spec is expected to be homogeneous across all nodes (machine),
    that is each node runs the same number of workers for a particular spec.

    Args:
        role: user-defined role for the workers with this spec
        local_world_size: number local workers to run
        fn: (deprecated use entrypoint instead)
        entrypoint: worker function or command
        args: arguments to pass to ``entrypoint``
        rdzv_handler: handles rdzv for this set of workers
        max_restarts: number of max retries for the workers
        monitor_interval: monitor status of workers every ``n`` seconds
        master_port: fixed port to run the c10d store on rank 0
                     if not specified then will chose a random free port
        master_addr: fixed master_addr to run the c10d store on rank 0
                     if not specified then will chose hostname on agent rank 0
        redirects: redirect std streams to a file,
                   selectively redirect for a particular
                   local rank by passing a map
        tee: tees the specified std stream(s) to console + file,
             selectively tee for a particular local rank by passing a map,
             takes precedence over ``redirects`` settings.

    """

    role: str
    local_world_size: int
    rdzv_handler: rdzv.RendezvousHandler
    fn: Optional[Callable] = None
    # TODO @kiuk - make entrypoint a required field
    entrypoint: Union[Callable, str, None] = None
    args: Tuple = ()
    max_restarts: int = 3
    monitor_interval: float = 0.1
    master_port: Optional[int] = None
    master_addr: Optional[str] = None
    local_addr: Optional[str] = None

    def __post_init__(self):
        assert self.local_world_size > 0
        assert self.monitor_interval > 0

        if self.fn:
            warnings.warn(
                "WorkerSpec.fn will be deprecated,"
                " please use WorkerSpec.entrypoint instead",
                category=DeprecationWarning,
            )
            self.entrypoint = self.fn
        assert self.entrypoint

    def get_entrypoint_name(self):
        """Get the entry point name.

        If the entrypoint is a function (e.g. ``Callable``) returns its ``__qualname__``
        else if the entrypoint is a binary (e.g. ``str``), returns the binary name.
        """
        if isinstance(self.entrypoint, str):
            return os.path.basename(self.entrypoint)
        else:
            assert self.entrypoint is not None
            return self.entrypoint.__qualname__

class WorkerGroup:
    """A set of ``Worker`` instances.

    The class defines a set of ``Worker`` instances for the given ``WorkerSpec`` managed by ``ElasticAgent``. Whether the worker
    group contains cross instance workers or not depends on the implementation of the agent.
    """

    __slots__ = [
        "spec",
        "workers",
        "store",
        "group_rank",
        "group_world_size",
        "state",
        "master_addr",
        "master_port",
    ]

    def __init__(self, spec: WorkerSpec):
        self.spec = spec
        self.workers = [Worker(local_rank=i) for i in range(self.spec.local_world_size)]

        # assigned after rdzv
        self.store = None
        self.group_rank = None
        self.group_world_size = None
        self.master_addr = None
        self.master_port = None

        self.state = WorkerState.INIT

