class PolynomialSparsityScheduler(SparsityScheduler):
    """
    Sparsity scheduler with a polynomial decay schedule.

    Two ways are available for calculations of the sparsity:
        - per epoch
        - per step
    Parameters `update_per_optimizer_step` and `steps_per_epoch`
    should be provided in config for the per step calculation.
    If `update_per_optimizer_step` was only provided then scheduler
    will use first epoch to calculate `steps_per_epoch`
    parameter. In this case, `current_epoch` and `current_step` will
    not be updated on this epoch. The scheduler will start calculation
    after `steps_per_epoch` will be calculated.
    """

    def __init__(self, controller: SparsityController, params: Dict[str, Any]):
        """
        Initializes a sparsity scheduler with a polynomial decay schedule.

        :param controller: Sparsity algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__(controller, params)
        self.schedule = PolynomialDecaySchedule(
            self.initial_level,
            self.target_level,
            self.target_epoch,
            params.get("power", SPARSITY_SCHEDULER_POWER),
            params.get("concave", SPARSITY_SCHEDULER_CONCAVE),
        )
        self._steps_in_current_epoch = 0
        self._update_per_optimizer_step = params.get(
            "update_per_optimizer_step", SPARSITY_SCHEDULER_UPDATE_PER_OPTIMIZER_STEP
        )
        self._steps_per_epoch = params.get("steps_per_epoch")
        self._should_skip = False

    def step(self, next_step: Optional[int] = None) -> None:
        self._steps_in_current_epoch += 1
        if self._should_skip:
            return

        super().step(next_step)
        if self._update_per_optimizer_step:
            self._update_sparsity_level()

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        self._maybe_should_skip()
        if self._should_skip:
            return

        self._steps_in_current_epoch = 0

        super().epoch_step(next_epoch)
        if not self._update_per_optimizer_step:
            self._update_sparsity_level()

    def _calculate_sparsity_level(self) -> float:
        local_step = max(self._steps_in_current_epoch - 1, 0)
        return self.schedule(self.current_epoch, local_step, self._steps_per_epoch)

    def load_state(self, state: Dict[str, Any]) -> None:
        super().load_state(state)
        if self._update_per_optimizer_step:
            self._steps_per_epoch = state["_steps_per_epoch"]

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        if self._update_per_optimizer_step:
            state["_steps_per_epoch"] = self._steps_per_epoch
        return state

    def _maybe_should_skip(self) -> None:
        """
        Checks if the first epoch (with index 0) should be skipped to calculate
        the steps per epoch. If the skip is needed, then the internal state
        of the scheduler object will not be changed.
        """
        self._should_skip = False
        if self._update_per_optimizer_step:
            if self._steps_per_epoch is None and self._steps_in_current_epoch > 0:
                self._steps_per_epoch = self._steps_in_current_epoch

            if (
                self._steps_per_epoch is not None
                and self._steps_in_current_epoch > 0
                and self._steps_per_epoch != self._steps_in_current_epoch
            ):
                raise Exception(
                    "Actual steps per epoch and steps per epoch from the scheduler "
                    "parameters are different. Scheduling may be incorrect."
                )

            if self._steps_per_epoch is None:
                self._should_skip = True
                nncf_logger.warning(
                    "Scheduler set to update sparsity level per optimizer step, "
                    "but steps_per_epoch was not set in config. Will only start updating "
                    "sparsity level after measuring the actual steps per epoch as signaled "
                    "by a .epoch_step() call."
                )

class AdaptiveSparsityScheduler(SparsityScheduler):
    """
    Sparsity scheduler with an adaptive schedule.
    """

    def __init__(self, controller: SparsityController, params: Dict[str, Any]):
        """
        Initializes a sparsity scheduler with an adaptive schedule.

        :param controller: Sparsity algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__(controller, params)
        self.decay_step = params.get("step", 0.05)
        self.eps = params.get("eps", 0.03)
        self.patience = params.get("patience", SPARSITY_SCHEDULER_PATIENCE)
        self.num_bad_epochs = 0
        self._current_level: float = self.initial_level

    @property
    def current_sparsity_level(self) -> float:
        """
        Returns sparsity level for the `current_epoch` or for step
        in the `current_epoch`.

        :return: Current sparsity level.
        """
        return self._current_level

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        super().epoch_step(next_epoch)
        self._update_sparsity_level()

    def _calculate_sparsity_level(self) -> float:
        if self._controller.current_sparsity_level >= self._current_level - self.eps:
            self.num_bad_epochs += 1

        current_level = self._current_level
        if self.num_bad_epochs >= self.patience:
            self.num_bad_epochs = 0
            current_level = current_level + self.decay_step

        self._current_level = min(current_level, self.target_level)

        return self._current_level

    def load_state(self, state: Dict[str, Any]) -> None:
        super().load_state(state)
        self.num_bad_epochs = state["num_bad_epochs"]
        self._current_level = state["current_sparsity_level"]

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["num_bad_epochs"] = self.num_bad_epochs
        state["current_sparsity_level"] = self._current_level
        return state

