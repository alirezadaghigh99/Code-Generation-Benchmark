class PrioritizedBuffer(Generic[T]):
    def __init__(
        self,
        capacity: Optional[int] = None,
        wait_priority_after_sampling: bool = True,
        initial_max_priority: float = 1.0,
    ):
        self.capacity = capacity
        self.data: Deque = collections.deque()
        self.priority_sums = SumTreeQueue()
        self.priority_mins = MinTreeQueue()
        self.max_priority = initial_max_priority
        self.wait_priority_after_sampling = wait_priority_after_sampling
        self.flag_wait_priority = False

    def __len__(self) -> int:
        return len(self.data)

    def append(self, value: T, priority: Optional[float] = None) -> None:
        if self.capacity is not None and len(self) == self.capacity:
            self.popleft()
        if priority is None:
            # Append with the highest priority
            priority = self.max_priority

        self.data.append(value)
        self.priority_sums.append(priority)
        self.priority_mins.append(priority)

    def popleft(self) -> T:
        assert len(self) > 0
        self.priority_sums.popleft()
        self.priority_mins.popleft()
        return self.data.popleft()

    def _sample_indices_and_probabilities(
        self, n: int, uniform_ratio: float
    ) -> Tuple[List[int], List[float], float]:
        total_priority: float = self.priority_sums.sum()
        min_prob = self.priority_mins.min() / total_priority
        indices = []
        priorities = []
        if uniform_ratio > 0:
            # Mix uniform samples and prioritized samples
            n_uniform = np.random.binomial(n, uniform_ratio)
            un_indices, un_priorities = self.priority_sums.uniform_sample(
                n_uniform, remove=self.wait_priority_after_sampling
            )
            indices.extend(un_indices)
            priorities.extend(un_priorities)
            n -= n_uniform
            min_prob = uniform_ratio / len(self) + (1 - uniform_ratio) * min_prob

        pr_indices, pr_priorities = self.priority_sums.prioritized_sample(
            n, remove=self.wait_priority_after_sampling
        )
        indices.extend(pr_indices)
        priorities.extend(pr_priorities)

        probs = [
            uniform_ratio / len(self) + (1 - uniform_ratio) * pri / total_priority
            for pri in priorities
        ]
        return indices, probs, min_prob

    def sample(
        self, n: int, uniform_ratio: float = 0
    ) -> Tuple[List[T], List[float], float]:
        """Sample data along with their corresponding probabilities.

        Args:
            n (int): Number of data to sample.
            uniform_ratio (float): Ratio of uniformly sampled data.
        Returns:
            sampled data (list)
            probabitilies (list)
        """
        assert not self.wait_priority_after_sampling or not self.flag_wait_priority
        indices, probabilities, min_prob = self._sample_indices_and_probabilities(
            n, uniform_ratio=uniform_ratio
        )
        sampled = [self.data[i] for i in indices]
        self.sampled_indices = indices
        self.flag_wait_priority = True
        return sampled, probabilities, min_prob

    def set_last_priority(self, priority: Sequence[float]) -> None:
        assert not self.wait_priority_after_sampling or self.flag_wait_priority
        assert all([p > 0.0 for p in priority])
        assert len(self.sampled_indices) == len(priority)
        for i, p in zip(self.sampled_indices, priority):
            self.priority_sums[i] = p
            self.priority_mins[i] = p
            self.max_priority = max(self.max_priority, p)
        self.flag_wait_priority = False
        self.sampled_indices = []

    def _uniform_sample_indices_and_probabilities(
        self, n: int
    ) -> Tuple[List[int], List[float]]:
        indices = list(sample_n_k(len(self.data), n))
        probabilities = [1 / len(self)] * len(indices)
        return indices, probabilities

