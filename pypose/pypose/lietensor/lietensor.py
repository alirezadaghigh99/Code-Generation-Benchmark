class Parameter(LieTensor, nn.Parameter):
    r'''
    A kind of LieTensor that is to be considered a module parameter.

    Parameters are of :meth:`LieTensor` and :meth:`torch.nn.Parameter`,
    that have a very special property when used with Modules: when
    they are assigned as Module attributes they are automatically
    added to the list of its parameters, and will appear, e.g., in
    :meth:`parameters()` iterator.

    Args:
        data (LieTensor): parameter LieTensor.
        requires_grad (bool, optional): if the parameter requires
            gradient. Default: ``True``

    Examples:
        >>> import torch, pypose as pp
        >>> x = pp.Parameter(pp.randn_SO3(2))
        >>> x.Log().sum().backward()
        >>> x.grad
        tensor([[0.8590, 1.4069, 0.6261, 0.0000],
                [1.2869, 1.0748, 0.5385, 0.0000]])
    '''
    def __init__(self, data, **kwargs):
        self.ltype = data.ltype

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.tensor([])
        return LieTensor._make_subclass(cls, data, requires_grad)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.clone(memory_format=torch.preserve_format))
            memo[id(self)] = result
            return result

