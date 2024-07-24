class Vindex:
    """
    Convenience wrapper around :func:`vindex`.

    The following are equivalent::

        Vindex(x)[..., i, j, :]
        vindex(x, (Ellipsis, i, j, slice(None)))

    :param torch.Tensor tensor: A tensor to be indexed.
    :return: An object with a special :meth:`__getitem__` method.
    """

    def __init__(self, tensor):
        self._tensor = tensor

    def __getitem__(self, args):
        return vindex(self._tensor, args)

class Index:
    """
    Convenience wrapper around :func:`index`.

    The following are equivalent::

        Index(x)[..., i, j, :]
        index(x, (Ellipsis, i, j, slice(None)))

    :param torch.Tensor tensor: A tensor to be indexed.
    :return: An object with a special :meth:`__getitem__` method.
    """

    def __init__(self, tensor):
        self._tensor = tensor

    def __getitem__(self, args):
        return index(self._tensor, args)

