class Progress:
    """Class to track progress during the loop. Includes state_dict/load_state_dict for convenience for checkpointing."""

    def __init__(
        self,
        num_epochs_completed: int = 0,
        num_steps_completed: int = 0,
        num_steps_completed_in_epoch: int = 0,
    ) -> None:
        self._num_epochs_completed: int = num_epochs_completed
        self._num_steps_completed: int = num_steps_completed
        self._num_steps_completed_in_epoch: int = num_steps_completed_in_epoch

    @property
    def num_epochs_completed(self) -> int:
        """Number of epochs completed thus far in loop."""
        return self._num_epochs_completed

    @property
    def num_steps_completed(self) -> int:
        """Number of steps completed thus far in loop."""
        return self._num_steps_completed

    @property
    def num_steps_completed_in_epoch(self) -> int:
        """Number of steps completed thus far in epoch."""
        return self._num_steps_completed_in_epoch

    def increment_step(self) -> None:
        """Increment the step counts completed and completed within the epoch."""
        self._num_steps_completed += 1
        self._num_steps_completed_in_epoch += 1

    def increment_epoch(self) -> None:
        """Increment the epochs completed and resets the steps completed within the epoch."""
        self._num_epochs_completed += 1
        self._num_steps_completed_in_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        """Returns a state_dict of a Progress instance in accordance with Stateful protocol."""
        return {
            "num_epochs_completed": self._num_epochs_completed,
            "num_steps_completed": self._num_steps_completed,
            "num_steps_completed_in_epoch": self._num_steps_completed_in_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores a Progress instance from a state_dict in accordance with Stateful protocol."""
        self._num_epochs_completed = state_dict["num_epochs_completed"]
        self._num_steps_completed = state_dict["num_steps_completed"]
        self._num_steps_completed_in_epoch = state_dict["num_steps_completed_in_epoch"]

    def get_progress_string(self) -> str:
        return f"completed epochs: {self.num_epochs_completed}, completed steps: {self.num_steps_completed}, completed steps in current epoch: {self.num_steps_completed_in_epoch}."

