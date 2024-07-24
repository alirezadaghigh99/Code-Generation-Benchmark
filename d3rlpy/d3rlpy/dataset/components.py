class Transition:
    r"""Transition tuple.

    Args:
        observation: Observation.
        action: Action
        reward: Reward. This could be a multi-step discounted return.
        next_observation: Observation at next timestep. This could be
            observation at multi-step ahead.
        next_action: Action at next timestep. This could be action at
            multi-step ahead.
        terminal: Flag of environment termination.
        interval: Timesteps between ``observation`` and ``next_observation``.
        rewards_to_go: Remaining rewards till the end of an episode, which is
            used to compute returns_to_go.
    """

    observation: Observation  # (...)
    action: NDArray  # (...)
    reward: Float32NDArray  # (1,)
    next_observation: Observation  # (...)
    next_action: NDArray  # (...)
    terminal: float
    interval: int
    rewards_to_go: Float32NDArray  # (L, 1)

    @property
    def observation_signature(self) -> Signature:
        r"""Returns observation sigunature.

        Returns:
            Observation signature.
        """
        shape = get_shape_from_observation(self.observation)
        dtype = get_dtype_from_observation(self.observation)
        if isinstance(self.observation, np.ndarray):
            shape = [shape]  # type: ignore
            dtype = [dtype]
        return Signature(dtype=dtype, shape=shape)  # type: ignore

    @property
    def action_signature(self) -> Signature:
        r"""Returns action signature.

        Returns:
            Action signature.
        """
        return Signature(
            dtype=[self.action.dtype],
            shape=[self.action.shape],
        )

    @property
    def reward_signature(self) -> Signature:
        r"""Returns reward signature.

        Returns:
            Reward signature.
        """
        return Signature(
            dtype=[self.reward.dtype],
            shape=[self.reward.shape],
        )

class Episode:
    r"""Standard episode implementation.

    Args:
        observations: Sequence of observations.
        actions: Sequence of actions.
        rewards: Sequence of rewards.
        terminated: Flag of environment termination.
    """

    observations: ObservationSequence
    actions: NDArray
    rewards: Float32NDArray
    terminated: bool

    @property
    def observation_signature(self) -> Signature:
        shape = get_shape_from_observation_sequence(self.observations)
        dtype = get_dtype_from_observation_sequence(self.observations)
        if isinstance(self.observations, np.ndarray):
            shape = [shape]  # type: ignore
            dtype = [dtype]
        return Signature(dtype=dtype, shape=shape)  # type: ignore

    @property
    def action_signature(self) -> Signature:
        return Signature(
            dtype=[self.actions.dtype],
            shape=[self.actions.shape[1:]],
        )

    @property
    def reward_signature(self) -> Signature:
        return Signature(
            dtype=[self.rewards.dtype],
            shape=[self.rewards.shape[1:]],
        )

    def size(self) -> int:
        return int(self.actions.shape[0])

    def compute_return(self) -> float:
        return float(np.sum(self.rewards))

    def serialize(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminated": self.terminated,
        }

    @classmethod
    def deserialize(cls, serializedData: Dict[str, Any]) -> "Episode":
        return cls(
            observations=serializedData["observations"],
            actions=serializedData["actions"],
            rewards=serializedData["rewards"],
            terminated=serializedData["terminated"],
        )

    def __len__(self) -> int:
        return self.actions.shape[0]

    @property
    def transition_count(self) -> int:
        return self.size() if self.terminated else self.size() - 1

class Signature:
    r"""Signature of arrays.

    Args:
        dtype: List of numpy data types.
        shape: List of array shapes.
    """

    dtype: Sequence[DType]
    shape: Sequence[Sequence[int]]

    def sample(self) -> Sequence[NDArray]:
        r"""Returns sampled arrays.

        Returns:
            List of arrays based on dtypes and shapes.
        """
        return [
            np.random.random(shape).astype(dtype)
            for shape, dtype in zip(self.shape, self.dtype)
        ]

class PartialTrajectory:
    r"""Partial trajectory.

    Args:
        observations: Sequence of observations.
        actions: Sequence of actions.
        rewards: Sequence of rewards.
        returns_to_go: Sequence of remaining returns.
        terminals: Sequence of terminal flags.
        timesteps: Sequence of timesteps.
        masks: Sequence of masks that represent padding.
        length: Sequence length.
    """

    observations: ObservationSequence  # (L, ...)
    actions: NDArray  # (L, ...)
    rewards: Float32NDArray  # (L, 1)
    returns_to_go: Float32NDArray  # (L, 1)
    terminals: Float32NDArray  # (L, 1)
    timesteps: Int32NDArray  # (L,)
    masks: Float32NDArray  # (L,)
    length: int

    @property
    def observation_signature(self) -> Signature:
        r"""Returns observation sigunature.

        Returns:
            Observation signature.
        """
        shape = get_shape_from_observation_sequence(self.observations)
        dtype = get_dtype_from_observation_sequence(self.observations)
        if isinstance(self.observations, np.ndarray):
            shape = [shape]  # type: ignore
            dtype = [dtype]
        return Signature(dtype=dtype, shape=shape)  # type: ignore

    @property
    def action_signature(self) -> Signature:
        r"""Returns action signature.

        Returns:
            Action signature.
        """
        return Signature(
            dtype=[self.actions.dtype],
            shape=[self.actions.shape[1:]],
        )

    @property
    def reward_signature(self) -> Signature:
        r"""Returns reward signature.

        Returns:
            Reward signature.
        """
        return Signature(
            dtype=[self.rewards.dtype],
            shape=[self.rewards.shape[1:]],
        )

    def get_transition_count(self) -> int:
        """Returns number of transitions.

        Returns:
            Number of transitions.
        """
        return self.length if bool(self.terminals[-1]) else self.length - 1

    def __len__(self) -> int:
        return self.length

