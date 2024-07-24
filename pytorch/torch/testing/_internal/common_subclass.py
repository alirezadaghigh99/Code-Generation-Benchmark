class SparseTensor(WrapperTensor):
    @classmethod
    def get_wrapper_properties(cls, size, values, indices, requires_grad=False):
        assert values.device == indices.device
        return values, {"size": size, "requires_grad": requires_grad}

    def __init__(self, size, values, indices, requires_grad=False):
        self.values = values
        self.indices = indices

    def __repr__(self):
        return super().__repr__(tensor_contents=f"values={self.values}, indices={self.indices}")

    def sparse_to_dense(self):
        res = torch.zeros(self.size(), dtype=self.values.dtype)
        res[self.indices.unbind(1)] = self.values
        return res

    @staticmethod
    def from_dense(t):
        indices = t.nonzero()
        values = t[indices.unbind(1)]
        return SparseTensor(t.size(), values, indices)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        func_name = f"{func.__module__}.{func.__name__}"

        res = cls._try_call_special_impl(func_name, args, kwargs)
        if res is not NotImplemented:
            return res

        # Otherwise, use a default implementation that construct dense
        # tensors and use that to compute values
        def unwrap(e):
            return e.sparse_to_dense() if isinstance(e, SparseTensor) else e

        # Wrap back all Tensors into our custom class
        def wrap(e):
            # Check for zeros and use that to get indices
            return SparseTensor.from_dense(e) if isinstance(e, torch.Tensor) else e

        rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
        return rs

    # To show how things happen later
    def __rmul__(self, other):
        return super().__rmul__(other)

    _SPECIAL_IMPLS = {}

    @classmethod
    def _try_call_special_impl(cls, func, args, kwargs):
        if func not in cls._SPECIAL_IMPLS:
            return NotImplemented
        return cls._SPECIAL_IMPLS[func](args, kwargs)

