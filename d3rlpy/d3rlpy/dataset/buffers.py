class InfiniteBuffer(BufferProtocol):
    r"""Buffer with unlimited capacity."""

    _transitions: List[Tuple[EpisodeBase, int]]
    _episodes: List[EpisodeBase]

    def __init__(self) -> None:
        self._transitions = []
        self._episodes = []
        self._transition_count = 0

    def append(self, episode: EpisodeBase, index: int) -> None:
        self._transitions.append((episode, index))
        if not self._episodes or episode is not self._episodes[-1]:
            self._episodes.append(episode)

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        return self._episodes

    @property
    def transition_count(self) -> int:
        return len(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions)

    def __getitem__(self, index: int) -> Tuple[EpisodeBase, int]:
        return self._transitions[index]

class FIFOBuffer(BufferProtocol):
    r"""FIFO buffer.

    Args:
        limit (int): buffer capacity.
    """

    _transitions: Deque[Tuple[EpisodeBase, int]]
    _episodes: List[EpisodeBase]
    _limit: int

    def __init__(self, limit: int):
        self._limit = limit
        self._transitions = deque(maxlen=limit)
        self._episodes = []

    def append(self, episode: EpisodeBase, index: int) -> None:
        if len(self._transitions) == self._limit:
            # check if dropped transition is the last transition
            if self._transitions[0][0] is not self._transitions[1][0]:
                self._episodes.pop(0)
        self._transitions.append((episode, index))
        if not self._episodes or episode is not self._episodes[-1]:
            self._episodes.append(episode)

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        return self._episodes

    @property
    def transition_count(self) -> int:
        return len(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions)

    def __getitem__(self, index: int) -> Tuple[EpisodeBase, int]:
        return self._transitions[index]

