def get_tensor_size_bytes_map(
    obj: object,
) -> Dict[torch.Tensor, int]:
    tensor_map = {}
    attributes_q = deque()
    attributes_q.append(obj)
    while attributes_q:
        attribute = attributes_q.popleft()
        if isinstance(attribute, torch.Tensor):
            tensor_map[attribute] = attribute.size().numel() * attribute.element_size()
        elif _is_named_tuple(attribute):
            attributes_q.extend(attribute._asdict().values())
        elif isinstance(attribute, Mapping):
            attributes_q.extend(attribute.values())
        elif isinstance(attribute, Sequence) and not isinstance(attribute, str):
            attributes_q.extend(attribute)
        elif hasattr(attribute, "__dict__") and not isinstance(attribute, Enum):
            attributes_q.extend(attribute.__dict__.values())
    return tensor_map

class RSSProfiler:
    """A profiler that periodically measures RSS (resident set size) delta.

    The baseline RSS is measured when the profiler is initialized.
    The RSS result is stored in the rss_deltas_bytes dict of the class.

    Attributes:
        interval: The interval for measuring RSS. The default value is 100ms.
        rss_deltas_bytes: The RSS delta bytes stored as dict. Key is the name for the profiling round, value is the list of RSS delta bytes captured periodically.
    """

    def __init__(self, interval: timedelta = _DEFAULT_MEASURE_INTERVAL) -> None:
        self.rss_deltas_bytes: Dict[str, List[int]] = {}
        self.interval = interval

    @contextmanager
    def profile(self, name: str) -> Generator[None, None, None]:
        """Profile the current process and store the results with a custom name as the key.

        Profile the process by starting a separate thread to capture the RSS periodically.
        The RSS result is stored in the rss_deltas_bytes dict of the class with the provided name as the key.

        Args:
            name: The name for the profiling round.
        """
        if name not in self.rss_deltas_bytes:
            self.rss_deltas_bytes[name] = []
        thread, stop_event = _get_target_thread(
            self.rss_deltas_bytes[name], self.interval
        )
        thread.start()
        try:
            yield
        finally:
            stop_event.set()
            thread.join()

    def reset(self) -> None:
        """
        Resets the stored rss_deltas_bytes dict to empty.
        """
        self.rss_deltas_bytes = {}

