class ProvenanceTensor(torch.Tensor):
    """
    Provenance tracking implementation in Pytorch.

    This class wraps a :class:`torch.Tensor` to track provenance through
    PyTorch ops, where provenance is a user-defined frozenset of objects. The
    provenance of the output tensors of any op is the union of provenances of
    input tensors.

    -   To start tracking provenance, wrap a :class:`torch.Tensor` in a
        :class:`ProvenanceTensor` with user-defined initial provenance.
    -   To read the provenance of a tensor use :meth:`get_provenance` .
    -   To detach provenance during a computation (similar to
        :meth:`~torch.Tensor.detach` to detach gradients during Pytorch
        computations), use the :meth:`detach_provenance` . This is useful to
        distinguish direct vs indirect provenance.

    Example::

        >>> a = ProvenanceTensor(torch.randn(3), frozenset({"a"}))
        >>> b = ProvenanceTensor(torch.randn(3), frozenset({"b"}))
        >>> c = torch.randn(3)
        >>> assert get_provenance(a + b + c) == frozenset({"a", "b"})
        >>> assert get_provenance(a + detach_provenance(b) + c) == frozenset({"a"})

    **References**

    [1] David Wingate, Noah Goodman, Andreas Stuhlmüller, Jeffrey Siskind (2011)
        Nonstandard Interpretations of Probabilistic Programs for Efficient Inference
        http://papers.neurips.cc/paper/4309-nonstandard-interpretations-of-probabilistic-programs-for-efficient-inference.pdf

    :param torch.Tensor data: An initial tensor to start tracking.
    :param frozenset provenance: An initial provenance set.
    """

    _t: torch.Tensor
    _provenance: frozenset

    def __new__(cls, data: torch.Tensor, provenance=frozenset(), **kwargs):
        assert not isinstance(data, ProvenanceTensor)
        if not provenance:
            return data
        ret = data.as_subclass(cls)
        ret._t = data  # this makes sure that detach_provenance always
        # returns the same object. This is important when
        # using the tensor as key in a dict, e.g. the global
        # param store
        ret._provenance = provenance
        return ret

    def __repr__(self):
        return "Provenance:\n{}\nTensor:\n{}".format(self._provenance, self._t)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        _args, _kwargs = detach_provenance([args, kwargs or {}])
        ret = func(*_args, **_kwargs)
        return track_provenance(ret, get_provenance([args, kwargs]))

