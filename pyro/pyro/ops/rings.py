class LogRing(Ring):
    """
    Ring of sum-product operations in log space.

    Tensor values are in log units, so ``sum`` is implemented as ``logsumexp``,
    and ``product`` is implemented as ``sum``.
    Tensor dimensions are packed; to read the name of a tensor, read the
    ``._pyro_dims`` attribute, which is a string of dimension names aligned
    with the tensor's shape.
    """

    _backend = "pyro.ops.einsum.torch_log"

    def __init__(self, cache=None, dim_to_size=None):
        super().__init__(cache=cache)
        self._dim_to_size = {} if dim_to_size is None else dim_to_size

    def sumproduct(self, terms, dims):
        inputs = [term._pyro_dims for term in terms]
        output = "".join(sorted(set("".join(inputs)) - set(dims)))
        equation = ",".join(inputs) + "->" + output
        term = contract(equation, *terms, backend=self._backend)
        term._pyro_dims = output
        return term

    def product(self, term, ordinal):
        dims = term._pyro_dims
        for dim in sorted(ordinal, reverse=True):
            pos = dims.find(dim)
            if pos != -1:
                key = "product", self._hash_by_id(term), dim
                if key in self._cache:
                    term = self._cache[key]
                else:
                    term = term.sum(pos)
                    dims = dims.replace(dim, "")
                    self._cache[key] = term
                    term._pyro_dims = dims
        return term

    def inv(self, term):
        key = "inv", self._hash_by_id(term)
        if key in self._cache:
            return self._cache[key]

        result = -term
        result = result.clamp(
            max=torch.finfo(result.dtype).max
        )  # avoid nan due to inf - inf
        result._pyro_dims = term._pyro_dims
        self._cache[key] = result
        return result

