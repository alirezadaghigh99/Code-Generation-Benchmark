def selection_config_from_dict(cfg: Dict[str, Any]) -> SelectionConfigV4:
    """Recursively converts selection config from dict to a SelectionConfigV4 instance."""
    strategies = []
    for entry in cfg.get("strategies", []):
        new_entry = copy.deepcopy(entry)
        new_entry["input"] = SelectionConfigV4EntryInput(**entry["input"])
        new_entry["strategy"] = SelectionConfigV4EntryStrategy(**entry["strategy"])
        strategies.append(SelectionConfigV4Entry(**new_entry))
    new_cfg = copy.deepcopy(cfg)
    new_cfg["strategies"] = strategies
    return SelectionConfigV4(**new_cfg)

class ComputeWorkerRunInfo:
    """Information about a Lightly Worker run.

    Attributes:
        state:
            The state of the Lightly Worker run.
        message:
            The last message of the Lightly Worker run.
    """

    state: Union[
        DockerRunState, DockerRunScheduledState.OPEN, STATE_SCHEDULED_ID_NOT_FOUND
    ]
    message: str

    def in_end_state(self) -> bool:
        """Checks whether the Lightly Worker run has ended."""
        return self.state in [
            DockerRunState.COMPLETED,
            DockerRunState.ABORTED,
            DockerRunState.FAILED,
            DockerRunState.CRASHED,
            STATE_SCHEDULED_ID_NOT_FOUND,
        ]

    def ended_successfully(self) -> bool:
        """Checkes whether the Lightly Worker run ended successfully or failed.

        Returns:
            A boolean value indicating if the Lightly Worker run was successful.
            True if the run was successful.

        Raises:
            ValueError:
                If the Lightly Worker run is still in progress.
        """
        if not self.in_end_state():
            raise ValueError("Lightly Worker run is still in progress.")
        return self.state == DockerRunState.COMPLETED

