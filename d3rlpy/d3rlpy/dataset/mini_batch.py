def from_transitions(
        cls, transitions: Sequence[Transition]
    ) -> "TransitionMiniBatch":
        r"""Constructs mini-batch from list of transitions.

        Args:
            transitions: List of transitions.

        Returns:
            Mini-batch.
        """
        observations = stack_observations(
            [transition.observation for transition in transitions]
        )
        actions = np.stack(
            [transition.action for transition in transitions], axis=0
        )
        rewards = np.stack(
            [transition.reward for transition in transitions], axis=0
        )
        next_observations = stack_observations(
            [transition.next_observation for transition in transitions]
        )
        next_actions = np.stack(
            [transition.next_action for transition in transitions], axis=0
        )
        terminals = np.reshape(
            np.array([transition.terminal for transition in transitions]),
            [-1, 1],
        )
        intervals = np.reshape(
            np.array([transition.interval for transition in transitions]),
            [-1, 1],
        )
        return TransitionMiniBatch(
            observations=cast_recursively(observations, np.float32),
            actions=cast_recursively(actions, np.float32),
            rewards=cast_recursively(rewards, np.float32),
            next_observations=cast_recursively(next_observations, np.float32),
            next_actions=cast_recursively(next_actions, np.float32),
            terminals=cast_recursively(terminals, np.float32),
            intervals=cast_recursively(intervals, np.float32),
            transitions=transitions,
        )

