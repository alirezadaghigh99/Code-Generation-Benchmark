class PTTargetPoint(TargetPoint):
    _OPERATION_TYPES = [
        TargetType.PRE_LAYER_OPERATION,
        TargetType.POST_LAYER_OPERATION,
        TargetType.OPERATION_WITH_WEIGHTS,
    ]
    _HOOK_TYPES = [TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATOR_POST_HOOK]

    _LAYER_TYPE = [TargetType.LAYER]

    _state_names = PTTargetPointStateNames

    def __init__(self, target_type: TargetType, target_node_name: NNCFNodeName, *, input_port_id: int = None):
        super().__init__(target_type)
        self.target_node_name = target_node_name
        self.target_type = target_type
        if self.target_type not in self._OPERATION_TYPES + self._HOOK_TYPES + self._LAYER_TYPE:
            raise NotImplementedError("Unsupported target type: {}".format(target_type))

        self.input_port_id = input_port_id

    def __eq__(self, other: "PTTargetPoint"):
        return (
            isinstance(other, PTTargetPoint)
            and self.target_type == other.target_type
            and self.target_node_name == other.target_node_name
            and self.input_port_id == other.input_port_id
        )

    def __str__(self):
        prefix = str(self.target_type)
        retval = prefix
        if self.target_type in self._OPERATION_TYPES + self._LAYER_TYPE:
            retval += " {}".format(self.target_node_name)
        elif self.target_type in self._HOOK_TYPES:
            if self.input_port_id is not None:
                retval += " {}".format(self.input_port_id)
            retval += " " + str(self.target_node_name)
        return retval

    def __hash__(self):
        return hash(str(self))

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            self._state_names.TARGET_TYPE: self.target_type.get_state(),
            self._state_names.INPUT_PORT: self.input_port_id,
            self._state_names.TARGET_NODE_NAME: self.target_node_name,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "PTTargetPoint":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        kwargs = {
            cls._state_names.TARGET_TYPE: TargetType.from_state(state[cls._state_names.TARGET_TYPE]),
            cls._state_names.INPUT_PORT: state[cls._state_names.INPUT_PORT],
            cls._state_names.TARGET_NODE_NAME: state[cls._state_names.TARGET_NODE_NAME],
        }
        return cls(**kwargs)

