class ScheduledModifierImpl(ScheduledModifier):
    def __init__(
        self,
        end_epoch: float = -1.0,
        start_epoch: float = -1.0,
    ):
        super().__init__()

