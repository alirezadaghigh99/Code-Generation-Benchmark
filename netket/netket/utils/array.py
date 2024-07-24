class HashableArray:
    """
    This class wraps a numpy or jax array in order to make it hashable and
    equality comparable (which is necessary since a well-defined hashable object
    needs to satisfy :code:`obj1 == obj2` whenever :code:`hash(obj1) == hash(obj2)`.

    The underlying array can also be accessed using :code:`numpy.asarray(self)`.
    """

    def __init__(self, wrapped: Array):
        """
        Wraps an array into an object that is hashable, and that can be
        converted again into an array.

        Forces all arrays to numpy and sets them to readonly.
        They can be converted back to jax later or a writeable numpy copy
        can be created by using `np.array(...)`

        The hash is computed by hashing the whole content of the array.

        Args:
            wrapped: array to be wrapped
        """
        if isinstance(wrapped, HashableArray):
            wrapped = wrapped.wrapped
        else:
            if isinstance(wrapped, jax.Array):
                # __array__ only works if it's a numpy array.
                wrapped = np.array(wrapped)
            else:
                wrapped = wrapped.copy()
            if isinstance(wrapped, np.ndarray):
                wrapped.flags.writeable = False

        self._wrapped: np.array = wrapped
        self._hash: Optional[int] = None

    @property
    def wrapped(self):
        """The read-only wrapped array."""
        return self._wrapped

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.wrapped.tobytes())
        return self._hash

    def __eq__(self, other):
        return (
            type(other) is HashableArray
            and self.shape == other.shape
            and self.dtype == other.dtype
            and hash(self) == hash(other)
        )

    def __array__(self, dtype: DType = None):
        if dtype is None:
            dtype = self.wrapped.dtype
        return self.wrapped.__array__(dtype)

    @property
    def dtype(self) -> DType:
        return self.wrapped.dtype

    @property
    def size(self) -> int:
        return self.wrapped.size

    @property
    def ndim(self) -> int:
        return self.wrapped.ndim

    @property
    def shape(self) -> Shape:
        return self.wrapped.shape

    def __repr__(self) -> str:
        return f"HashableArray({self.wrapped},\n shape={self.shape}, dtype={self.dtype}, hash={hash(self)})"

    def __str__(self) -> str:
        return (
            f"HashableArray(shape={self.shape}, dtype={self.dtype}, hash={hash(self)})"
        )

