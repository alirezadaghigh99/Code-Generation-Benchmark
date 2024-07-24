class ReplayBuffer(ReplayBufferBase):
    r"""Replay buffer for experience replay.

    This replay buffer implementation is used for both online and offline
    training in d3rlpy. To determine shapes of observations, actions and
    rewards, one of ``episodes``, ``env`` and signatures must be provided.

    .. code-block::

        from d3rlpy.dataset import FIFOBuffer, ReplayBuffer, Signature

        buffer = FIFOBuffer(limit=1000000)

        # initialize with pre-collected episodes
        replay_buffer = ReplayBuffer(buffer=buffer, episodes=<episodes>)

        # initialize with Gym
        replay_buffer = ReplayBuffer(buffer=buffer, env=<env>)

        # initialize with manually specified signatures
        replay_buffer = ReplayBuffer(
            buffer=buffer,
            observation_signature=Signature(dtype=[<dtype>], shape=[<shape>]),
            action_signature=Signature(dtype=[<dtype>], shape=[<shape>]),
            reward_signature=Signature(dtype=[<dtype>], shape=[<shape>]),
        )

    Args:
        buffer (d3rlpy.dataset.BufferProtocol): Buffer implementation.
        transition_picker (Optional[d3rlpy.dataset.TransitionPickerProtocol]):
            Transition picker implementation for Q-learning-based algorithms.
            If ``None`` is given, ``BasicTransitionPicker`` is used by default.
        trajectory_slicer (Optional[d3rlpy.dataset.TrajectorySlicerProtocol]):
            Trajectory slicer implementation for Transformer-based algorithms.
            If ``None`` is given, ``BasicTrajectorySlicer`` is used by default.
        writer_preprocessor (Optional[d3rlpy.dataset.WriterPreprocessProtocol]):
            Writer preprocessor implementation. If ``None`` is given,
            ``BasicWriterPreprocess`` is used by default.
        episodes (Optional[Sequence[d3rlpy.dataset.EpisodeBase]]):
            List of episodes to initialize replay buffer.
        env (Optional[GymEnv]): Gym environment to extract shapes of
            observations and action.
        observation_signature (Optional[d3rlpy.dataset.Signature]):
            Signature of observation.
        action_signature (Optional[d3rlpy.dataset.Signature]):
            Signature of action.
        reward_signature (Optional[d3rlpy.dataset.Signature]):
            Signature of reward.
        action_space (Optional[d3rlpy.constants.ActionSpace]):
            Action-space type.
        action_size (Optional[int]): Size of action-space. For continuous
            action-space, this represents dimension of action vectors. For
            discrete action-space, this represents the number of discrete
            actions.
        cache_size (int): Size of cache to record active episode history used
            for online training. ``cache_size`` needs to be greater than the
            maximum possible episode length.
        write_at_termination (bool): Flag to write experiences to the buffer at the
            end of an episode all at once.
    """

    _buffer: BufferProtocol
    _transition_picker: TransitionPickerProtocol
    _trajectory_slicer: TrajectorySlicerProtocol
    _writer: ExperienceWriter
    _episodes: List[EpisodeBase]
    _dataset_info: DatasetInfo

    def __init__(
        self,
        buffer: BufferProtocol,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
        episodes: Optional[Sequence[EpisodeBase]] = None,
        env: Optional[GymEnv] = None,
        observation_signature: Optional[Signature] = None,
        action_signature: Optional[Signature] = None,
        reward_signature: Optional[Signature] = None,
        action_space: Optional[ActionSpace] = None,
        action_size: Optional[int] = None,
        cache_size: int = 10000,
        write_at_termination: bool = False,
    ):
        transition_picker = transition_picker or BasicTransitionPicker()
        trajectory_slicer = trajectory_slicer or BasicTrajectorySlicer()
        writer_preprocessor = writer_preprocessor or BasicWriterPreprocess()

        if not (
            observation_signature and action_signature and reward_signature
        ):
            if episodes:
                observation_signature = episodes[0].observation_signature
                action_signature = episodes[0].action_signature
                reward_signature = episodes[0].reward_signature
            elif env:
                observation_signature = Signature(
                    dtype=[env.observation_space.dtype],
                    shape=[env.observation_space.shape],  # type: ignore
                )
                action_signature = Signature(
                    dtype=[env.action_space.dtype],
                    shape=[env.action_space.shape],  # type: ignore
                )
                reward_signature = Signature(
                    dtype=[np.dtype(np.float32)],
                    shape=[[1]],
                )
            else:
                raise ValueError(
                    "Either episodes or env must be provided to determine signatures."
                    " Or specify signatures directly."
                )
            LOG.info(
                "Signatures have been automatically determined.",
                observation_signature=observation_signature,
                action_signature=action_signature,
                reward_signature=reward_signature,
            )

        if action_space is None:
            if episodes:
                action_space = detect_action_space(episodes[0].actions)
            elif env:
                action_space = detect_action_space_from_env(env)
            else:
                raise ValueError(
                    "Either episodes or env must be provided to determine action_space."
                    " Or specify action_space directly."
                )
            LOG.info(
                "Action-space has been automatically determined.",
                action_space=action_space,
            )

        if action_size is None:
            if episodes:
                if action_space == ActionSpace.CONTINUOUS:
                    action_size = action_signature.shape[0][0]
                else:
                    max_action = 0
                    for episode in episodes:
                        max_action = max(
                            int(np.max(episode.actions)), max_action
                        )
                    action_size = max_action + 1  # index should start from 0
            elif env:
                action_size = detect_action_size_from_env(env)
            else:
                raise ValueError(
                    "Either episodes or env must be provided to determine action_space."
                    " Or specify action_size directly."
                )
            LOG.info(
                "Action size has been automatically determined.",
                action_size=action_size,
            )

        self._buffer = buffer
        self._writer = ExperienceWriter(
            buffer,
            writer_preprocessor,
            observation_signature=observation_signature,
            action_signature=action_signature,
            reward_signature=reward_signature,
            cache_size=cache_size,
            write_at_termination=write_at_termination,
        )
        self._transition_picker = transition_picker
        self._trajectory_slicer = trajectory_slicer
        self._dataset_info = DatasetInfo(
            observation_signature=observation_signature,
            action_signature=action_signature,
            reward_signature=reward_signature,
            action_space=action_space,
            action_size=action_size,
        )

        if episodes:
            for episode in episodes:
                self.append_episode(episode)

    def append(
        self,
        observation: Observation,
        action: Union[int, NDArray],
        reward: Union[float, NDArray],
    ) -> None:
        self._writer.write(observation, action, reward)

    def append_episode(self, episode: EpisodeBase) -> None:
        for i in range(episode.transition_count):
            self._buffer.append(episode, i)

    def clip_episode(self, terminated: bool) -> None:
        self._writer.clip_episode(terminated)

    def sample_transition(self) -> Transition:
        index = np.random.randint(self._buffer.transition_count)
        episode, transition_index = self._buffer[index]
        return self._transition_picker(episode, transition_index)

    def sample_transition_batch(self, batch_size: int) -> TransitionMiniBatch:
        return TransitionMiniBatch.from_transitions(
            [self.sample_transition() for _ in range(batch_size)]
        )

    def sample_trajectory(self, length: int) -> PartialTrajectory:
        index = np.random.randint(self._buffer.transition_count)
        episode, transition_index = self._buffer[index]
        return self._trajectory_slicer(episode, transition_index, length)

    def sample_trajectory_batch(
        self, batch_size: int, length: int
    ) -> TrajectoryMiniBatch:
        return TrajectoryMiniBatch.from_partial_trajectories(
            [self.sample_trajectory(length) for _ in range(batch_size)]
        )

    def dump(self, f: BinaryIO) -> None:
        dump(self._buffer.episodes, f)

    @classmethod
    def from_episode_generator(
        cls,
        episode_generator: EpisodeGeneratorProtocol,
        buffer: BufferProtocol,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    ) -> "ReplayBuffer":
        return cls(
            buffer,
            episodes=episode_generator(),
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            writer_preprocessor=writer_preprocessor,
        )

    @classmethod
    def load(
        cls,
        f: BinaryIO,
        buffer: BufferProtocol,
        episode_cls: Type[EpisodeBase] = Episode,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    ) -> "ReplayBuffer":
        return cls(
            buffer,
            episodes=load(episode_cls, f),
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            writer_preprocessor=writer_preprocessor,
        )

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        return self._buffer.episodes

    def size(self) -> int:
        return len(self._buffer.episodes)

    @property
    def buffer(self) -> BufferProtocol:
        return self._buffer

    @property
    def transition_count(self) -> int:
        return self._buffer.transition_count

    @property
    def transition_picker(self) -> TransitionPickerProtocol:
        return self._transition_picker

    @property
    def trajectory_slicer(self) -> TrajectorySlicerProtocol:
        return self._trajectory_slicer

    @property
    def dataset_info(self) -> DatasetInfo:
        return self._dataset_info

