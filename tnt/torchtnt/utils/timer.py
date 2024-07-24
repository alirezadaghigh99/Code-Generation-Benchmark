def get_synced_durations_histogram(
    recorded_durations: Dict[str, List[float]],
    percentiles: Sequence[float],
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[str, float]]:
    """Synchronizes the recorded durations across ranks.

    Args:
        recorded_durations: The mapping of durations to sync and compute histograms from.
        percentiles: The percentiles to compute. Values should be in the range [0, 100].
        pg (optional): The process group to use for synchronization. Defaults to the global process group.

    Returns:
        A dictionary mapping the action names to a dictionary of the computed percentiles, along with the mean duration of each action.

    Raises:
        ValueError: If the input percentiles are not in the range [0, 100].
    """
    _validate_percentiles(percentiles)
    synced_durations = _sync_durations(recorded_durations, pg)
    return get_durations_histogram(synced_durations, percentiles=percentiles)

class Timer(TimerProtocol):
    def __init__(
        self,
        *,
        cuda_sync: Optional[bool] = None,
        verbose: bool = False,
    ) -> None:
        """
        A Timer class which implements TimerProtocol and stores timings in a dictionary `recorded_durations`.

        Args:
            cuda_sync: whether to call torch.cuda.synchronize() before and after timing. Defaults to True if CUDA is available.
            verbose: whether to enable verbose logging.

        Note:
            Enabling cuda_sync will incur a performance hit, but will ensure accurate timings on GPUs.

        Raises:
            ValueError: If cuda_sync is set to True but CUDA is not available.

        """
        if cuda_sync and not torch.cuda.is_available():
            raise ValueError(
                "CUDA must be available in order to enable CUDA synchronization."
            )
        self.cuda_sync: bool = (
            cuda_sync if cuda_sync is not None else torch.cuda.is_available()
        )
        self.verbose = verbose
        self.recorded_durations: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def time(
        self,
        action_name: str,
    ) -> Generator[None, None, None]:
        """
        A context manager for timing a code block, with optional cuda synchronization and verbose timing.

        Args:
            action_name: the name under which to store the timing of what is enclosed in the context manager.
        """
        start_time: float = perf_counter()
        try:
            if self.cuda_sync:
                torch.cuda.synchronize()
            yield
        finally:
            if self.cuda_sync:
                torch.cuda.synchronize()
            interval_time: float = perf_counter() - start_time
            if self.verbose:
                logger.info(f"{action_name} took {interval_time} seconds.")
        self.recorded_durations[action_name].append(interval_time)

    def reset(self) -> None:
        """
        Reset the recorded_durations to an empty list
        """
        self.recorded_durations = defaultdict(list)

class FullSyncPeriodicTimer:
    """
    Measures time (resets if given interval elapses) on rank 0
    and propagates result to other ranks.
    Propagation is done asynchronously from previous step
    in order to avoid blocking of a training process.
    """

    def __init__(self, interval: datetime.timedelta, cpu_pg: dist.ProcessGroup) -> None:
        self._interval = interval
        self._cpu_pg = cpu_pg
        self._prev_time: float = perf_counter()
        self._timeout_tensor: torch.Tensor = torch.zeros(1, dtype=torch.int)
        self._prev_work: Optional[Work] = None

    def check(self) -> bool:
        ret = False
        curr_time = perf_counter()

        if self._prev_work is not None:
            self._prev_work.wait()
            ret = self._timeout_tensor[0].item() == 1
            if ret:
                self._prev_time = curr_time

        self._timeout_tensor[0] = (
            1 if (curr_time - self._prev_time) >= self._interval.total_seconds() else 0
        )
        self._prev_work = dist.broadcast(
            self._timeout_tensor, 0, group=self._cpu_pg, async_op=True
        )

        return ret

    def wait_remaining_work(self) -> None:
        if self._prev_work is not None:
            self._prev_work.wait()

class BoundedTimer(Timer):
    """
    A Timer class which implements TimerProtocol and stores timings in a dictionary `recorded_durations`.

    Same behavior as timer, but with the addition of size_bounds = (lower, upper)

    Args:
        ...
        size_bounds: defines the range of samples that should be kept in the timer. The lower bound should be smaller than
            the upper bound. When the number of samples reaches the upper bound, the oldest (upper-lower) bound samples will
            be removed. This range is applied per action.
    """

    def __init__(self, lower_bound: int, upper_bound: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert lower_bound > 0
        assert lower_bound < upper_bound
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @contextmanager
    def time(
        self,
        action_name: str,
    ) -> Generator[None, None, None]:
        with super().time(action_name):
            yield
        self._apply_bounds(action_name)

    def _apply_bounds(self, action_name: str) -> None:
        # Keep 'lower_bound' most recent samples, if at or over upper bound
        n_samples: int = len(self.recorded_durations[action_name])
        if self.upper_bound <= n_samples:
            self.recorded_durations[action_name] = list(
                self.recorded_durations[action_name][-self.lower_bound :]
            )

