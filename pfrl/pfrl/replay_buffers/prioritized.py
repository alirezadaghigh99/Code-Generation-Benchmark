class PrioritizedReplayBuffer(ReplayBuffer, PriorityWeightError):
    """Stochastic Prioritization

    https://arxiv.org/pdf/1511.05952.pdf Section 3.3
    proportional prioritization

    Args:
        capacity (int): capacity in terms of number of transitions
        alpha (float): Exponent of errors to compute probabilities to sample
        beta0 (float): Initial value of beta
        betasteps (int): Steps to anneal beta to 1
        eps (float): To revisit a step after its error becomes near zero
        normalize_by_max (bool): Method to normalize weights. ``'batch'`` or
            ``True`` (default): divide by the maximum weight in the sampled
            batch. ``'memory'``: divide by the maximum weight in the memory.
            ``False``: do not normalize
    """

    def __init__(
        self,
        capacity=None,
        alpha=0.6,
        beta0=0.4,
        betasteps=2e5,
        eps=0.01,
        normalize_by_max=True,
        error_min=0,
        error_max=1,
        num_steps=1,
    ):
        self.capacity = capacity
        assert num_steps > 0
        self.num_steps = num_steps
        self.memory = PrioritizedBuffer(capacity=capacity)
        self.last_n_transitions = collections.defaultdict(
            lambda: collections.deque([], maxlen=num_steps)
        )
        PriorityWeightError.__init__(
            self,
            alpha,
            beta0,
            betasteps,
            eps,
            normalize_by_max,
            error_min=error_min,
            error_max=error_max,
        )

    def sample(self, n):
        assert len(self.memory) >= n
        sampled, probabilities, min_prob = self.memory.sample(n)
        weights = self.weights_from_probabilities(probabilities, min_prob)
        for e, w in zip(sampled, weights):
            e[0]["weight"] = w
        return sampled

    def update_errors(self, errors):
        self.memory.set_last_priority(self.priority_from_errors(errors))