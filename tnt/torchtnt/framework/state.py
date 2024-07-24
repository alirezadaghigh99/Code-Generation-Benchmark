class PhaseState(Generic[TData, TStepOutput]):
    """State for each phase (train, eval, predict).
    Modified by the framework, read-only for the user.
    """

    def __init__(
        self,
        *,
        dataloader: Iterable[TData],
        max_epochs: Optional[int] = None,  # used only for train
        max_steps: Optional[int] = None,  # used only for train
        max_steps_per_epoch: Optional[int] = None,
        evaluate_every_n_steps: Optional[int] = None,  # used only for evaluate
        evaluate_every_n_epochs: Optional[int] = None,  # used only for evaluate
    ) -> None:
        _check_loop_condition("max_epochs", max_epochs)
        _check_loop_condition("max_steps", max_steps)
        _check_loop_condition("max_steps_per_epoch", max_steps_per_epoch)
        _check_loop_condition("evaluate_every_n_steps", evaluate_every_n_steps)
        _check_loop_condition("evaluate_every_n_epochs", evaluate_every_n_epochs)

        self._dataloader: Iterable[TData] = dataloader
        self._max_epochs = max_epochs
        self._max_steps = max_steps
        self._max_steps_per_epoch = max_steps_per_epoch
        self._evaluate_every_n_steps = evaluate_every_n_steps
        self._evaluate_every_n_epochs = evaluate_every_n_epochs

        self._step_output: Optional[TStepOutput] = None
        self._iteration_timer = BoundedTimer(
            cuda_sync=False, lower_bound=1_000, upper_bound=5_000
        )

    @property
    def dataloader(self) -> Iterable[TData]:
        """Dataloader defined by the user."""
        return self._dataloader

    @property
    def max_epochs(self) -> Optional[int]:
        """Maximum number of epochs to train, defined by the user."""
        return self._max_epochs

    @property
    def max_steps(self) -> Optional[int]:
        """Maximum number of steps to train, defined by the user."""
        return self._max_steps

    @property
    def max_steps_per_epoch(self) -> Optional[int]:
        """Maximum number of steps to run per epoch, defined by the user."""
        return self._max_steps_per_epoch

    @property
    def evaluate_every_n_steps(self) -> Optional[int]:
        """Frequency with which to evaluate in terms of training steps, when running :func:`~torchtnt.framework.fit`. Defined by the user."""
        return self._evaluate_every_n_steps

    @property
    def evaluate_every_n_epochs(self) -> Optional[int]:
        """Frequency with which to evaluate in terms of training epochs, when running :func:`~torchtnt.framework.fit`. Defined by the user."""
        return self._evaluate_every_n_epochs

    @property
    def step_output(self) -> Optional[TStepOutput]:
        """Output of the last step."""
        return self._step_output

    @property
    def iteration_timer(self) -> TimerProtocol:
        """An always-on :class:`~torchtnt.utils.TimerProtocol` object which contains CPU timings (without synchronisation) of the iterations."""
        return self._iteration_timer

