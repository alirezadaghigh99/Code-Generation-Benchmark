class Dim(_C.Dim, _Tensor):
    # note that _C.Dim comes before tensor because we want the Dim API for things like size to take precendence.
    # Tensor defines format, but we want to print Dims with special formatting
    __format__ = object.__format__class Tensor(_Tensor, _C.Tensor):
    if not use_c:
        from_batched = staticmethod(_C.Tensor_from_batched)
    from_positional = staticmethod(_C.Tensor_from_positional)
    sum = _C._instancemethod(_C.Tensor_sum)    def dims(self):
        return tuple(d for d in self._levels if isinstance(d, Dim))