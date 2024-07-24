class Permutation(Element):
    def __init__(self, permutation: Array, name: Optional[str] = None):
        r"""
        Creates a `Permutation` from an array of preimages of :code:`range(N)`.

        Arguments:
            permutation: 1D array listing :math:`g^{-1}(x)` for all :math:`0\le x < N`
                (i.e., :code:`V[permutation]` permutes the elements of `V` as desired)
            name: optional, custom name for the permutation

        Returns:
            a `Permutation` object encoding the same permutation
        """
        self.permutation = HashableArray(np.asarray(permutation))
        self.__name = name

    def __hash__(self):
        return hash(self.permutation)

    def __eq__(self, other):
        if isinstance(other, Permutation):
            return self.permutation == other.permutation
        else:
            return False

    @property
    def _name(self):
        return self.__name

    def __repr__(self):
        if self._name is not None:
            return self._name
        else:
            return f"Permutation({np.asarray(self).tolist()})"

    def __array__(self, dtype: DType = None):
        return np.asarray(self.permutation, dtype)

    def apply_to_id(self, x: Array):
        """Returns the image of indices `x` under the permutation"""
        return np.argsort(self.permutation)[x]

