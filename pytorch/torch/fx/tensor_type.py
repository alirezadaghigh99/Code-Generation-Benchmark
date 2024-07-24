class TensorType:
    """
    TensorType defines a type for tensors, which consists of a list of dimensions.
    Example:
        class M(torch.nn.Module):
            def forward(self, x:TensorType((1,2,3, Dyn)), y:TensorType((1,2,3, Dyn))):
                return torch.add(x, y)
    """

    def __init__(self, dim):
        self.__origin__ = TensorType
        self.__args__ = dim

    def __repr__(self):
        return f'TensorType[{self.__args__}]'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return list(self.__args__) == list(other.__args__)
        else:
            return False

    @staticmethod
    def __class_getitem__(*args):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        return TensorType(tuple(args))

