class TFLayer(TFLayerPoint):
    """
    `TFLayer` defines a layer in the TensorFlow model graph.

    For example, `TFLayer` is used to specify the layer in the removal command
    to remove from the model.
    """

    _state_names = TFLayerStateNames

    def __init__(self, layer_name: str):
        super().__init__(TargetType.LAYER, layer_name)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            self._state_names.LAYER_NAME: self.layer_name,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TFLayer":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)

class TFMultipleInsertionCommands(TFTransformationCommand):
    """
    A list of insertion commands combined by a common global target point but
    with different target points in between.

    For example, If a layer has multiple weight variables you can use this
    transformation command to insert operations with weights for each layer
    weights variable at one multiple insertion command.
    """

    def __init__(
        self,
        target_point: TargetPoint,
        check_target_points_fn: Optional[Callable] = None,
        commands: Optional[List[TFTransformationCommand]] = None,
    ):
        super().__init__(TransformationType.MULTI_INSERT, target_point)
        self.check_target_points_fn = check_target_points_fn
        if check_target_points_fn is None:
            self.check_target_points_fn = lambda tp0, tp1: tp0 == tp1
        self._commands = []
        if commands is not None:
            for cmd in commands:
                self.add_insertion_command(cmd)

    @property
    def commands(self) -> List[TFTransformationCommand]:
        return self._commands

    def check_insertion_command(self, command: TFTransformationCommand) -> bool:
        if (
            isinstance(command, TFTransformationCommand)
            and command.type == TransformationType.INSERT
            and self.check_target_points_fn(self.target_point, command.target_point)
        ):
            return True
        return False

    def add_insertion_command(self, command: TFTransformationCommand) -> None:
        if not self.check_insertion_command(command):
            raise ValueError("{} command could not be added".format(type(command).__name__))

        for idx, cmd in enumerate(self.commands):
            if cmd.target_point == command.target_point:
                self.commands[idx] = cmd + command
                break
        else:
            self.commands.append(command)

    def union(self, other: TFTransformationCommand) -> "TFMultipleInsertionCommands":
        if not self.check_command_compatibility(other):
            raise ValueError("{} and {} commands could not be united".format(type(self).__name__, type(other).__name__))

        def make_check_target_points_fn(fn1, fn2):
            def check_target_points(tp0, tp1):
                return fn1(tp0, tp1) or fn2(tp0, tp1)

            return check_target_points

        check_target_points_fn = (
            self.check_target_points_fn
            if self.check_target_points_fn == other.check_target_points_fn
            else make_check_target_points_fn(self.check_target_points_fn, other.check_target_points_fn)
        )

        multi_cmd = TFMultipleInsertionCommands(self.target_point, check_target_points_fn, self.commands)
        for cmd in other.commands:
            multi_cmd.add_insertion_command(cmd)
        return multi_cmd

class TFRemovalCommand(TFTransformationCommand):
    """
    Removes the target object.
    """

    def __init__(self, target_point: TargetPoint):
        super().__init__(TransformationType.REMOVE, target_point)

    def union(self, other: TFTransformationCommand) -> "TFRemovalCommand":
        raise NotImplementedError("A command of TFRemovalCommand type could not be united with another command")

