class PrioritizedReplayBuffer(MultiStepReplayBuffer):
    """The Prioritized Experience Replay Buffer class. Used to store experiences and allow
    off-policy learning.

    :param memory_size: Maximum length of replay buffer
    :type memory_size: int
    :param field_names: Field names for experience named tuple, e.g. ['state', 'action', 'reward']
    :type field_names: list[str]
    :param num_envs: Number of parallel environments for training
    :type num_envs: int
    :param alpha: Alpha parameter for prioritized replay buffer, defaults to 0.6
    :type alpha: float, optional
    :param n_step: Step number to calculate n-step td error, defaults to 1
    :type n_step: int, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to None
    :type device: str, optional
    """

    def __init__(
        self,
        memory_size,
        field_names,
        num_envs,
        alpha=0.6,
        n_step=1,
        gamma=0.99,
        device=None,
    ):
        super().__init__(memory_size, field_names, num_envs, n_step, gamma, device)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.memory_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def _add(self, *args):
        super()._add(*args)
        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.memory_size

    def sample(self, batch_size, beta=0.4):
        """Returns sample of experiences from memory.

        :param batch_size: Number of samples to return
        :type batch_size: int
        """
        idxs = self._sample_proprtional(batch_size)
        experiences = [self.memory[i] for i in idxs]
        transition = self._process_transition(experiences)

        weights = torch.from_numpy(
            np.array([self._calculate_weight(i, beta) for i in idxs])
        ).float()

        if self.device is not None:
            weights = weights.to(self.device)

        transition["weights"] = weights
        transition["idxs"] = idxs

        return tuple(transition.values())

    def update_priorities(self, idxs, priorities):
        """Update priorities of sampled transitions."""
        for idx, priority in zip(idxs, priorities):
            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)

    def _sample_proprtional(self, batch_size):
        """Sample indices based on proportions.

        :param batch_size: Sample size
        :type batch_size: int
        """
        idxs = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            idxs.append(idx)
        return idxs

    def _calculate_weight(self, idx, beta):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight