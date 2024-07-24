class MultiStateful:
    """
    Wrapper for multiple stateful objects. Necessary because we might have multiple nn.Modules or multiple optimizers,
    but save/load_checkpoint APIs may only accepts one stateful object.

    Stores state_dict as a dict of state_dicts.
    """

    def __init__(
        self,
        stateful_objs: Union[
            StatefulDict, ModuleDict, OptimizerAndLRSchedulerDict, ProgressDict
        ],
    ) -> None:
        self.stateful_objs = stateful_objs

    def state_dict(self) -> Dict[str, Any]:
        return {k: v.state_dict() for k, v in self.stateful_objs.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for k in state_dict:
            self.stateful_objs[k].load_state_dict(state_dict[k])

