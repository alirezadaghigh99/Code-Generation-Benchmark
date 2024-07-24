class BuildingBlock:
    """
    Describes a building block that is uniquely defined by the start and end nodes.
    """

    def __init__(self, start_node_name: NNCFNodeName, end_node_name: NNCFNodeName):
        self.start_node_name = start_node_name
        self.end_node_name = end_node_name

    def __eq__(self, __o: "BuildingBlock") -> bool:
        return self.start_node_name == __o.start_node_name and self.end_node_name == __o.end_node_name

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "[START NODE: {}, END_NODE: {}]".format(self.start_node_name, self.end_node_name)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {"start_node_name": self.start_node_name, "end_node_name": self.end_node_name}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "BuildingBlock":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return BuildingBlock(**state)

