class PrepareNode(Operation):
    """Placeholder node for state preparations"""

    num_wires = 1
    grad_method = None
    num_params = 0

    def __init__(self, wires=None, id=None):
        id = id or str(uuid.uuid4())

        super().__init__(wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        op_label = base_label or self.__class__.__name__
        return op_label

class MeasureNode(Operation):
    """Placeholder node for measurement operations"""

    num_wires = 1
    grad_method = None
    num_params = 0

    def __init__(self, wires=None, id=None):
        id = id or str(uuid.uuid4())

        super().__init__(wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        op_label = base_label or self.__class__.__name__
        return op_label

