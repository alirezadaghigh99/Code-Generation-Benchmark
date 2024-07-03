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
        return self.state == WorkerState.FAILEDdef _get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())