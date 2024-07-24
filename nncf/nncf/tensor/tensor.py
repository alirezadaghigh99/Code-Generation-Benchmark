class Tensor:
    """
    An interface to framework specific tensors for common NNCF algorithms.
    """

    def __init__(self, data: Optional[TTensor]):
        self._data = data.data if isinstance(data, Tensor) else data

    @property
    def data(self) -> TTensor:
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def device(self) -> TensorDeviceType:
        return _call_function("device", self)

    @property
    def dtype(self) -> TensorDataType:
        return _call_function("dtype", self)

    @property
    def backend(self) -> TensorBackend:
        return _call_function("backend", self)

    @property
    def size(self) -> int:
        return _call_function("size", self)

    def __bool__(self) -> bool:
        return bool(self.data)

    def __iter__(self):
        return TensorIterator(self.data)

    def __getitem__(self, index: Union[Tensor, int, Tuple[Union[Tensor, int], ...]]) -> Tensor:
        return Tensor(self.data[unwrap_index(index)])

    def __setitem__(self, index: Union[Tensor, int, Tuple[Union[Tensor, int], ...]], value: Any) -> None:
        self.data[unwrap_index(index)] = unwrap_tensor_data(value)

    def __str__(self) -> str:
        return f"nncf.Tensor({str(self.data)})"

    def __repr__(self) -> str:
        return f"nncf.Tensor({repr(self.data)})"

    # built-in operations

    def __add__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data + unwrap_tensor_data(other))

    def __radd__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(unwrap_tensor_data(other) + self.data)

    def __sub__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data - unwrap_tensor_data(other))

    def __rsub__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(unwrap_tensor_data(other) - self.data)

    def __mul__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data * unwrap_tensor_data(other))

    def __rmul__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(unwrap_tensor_data(other) * self.data)

    def __pow__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data ** unwrap_tensor_data(other))

    def __rpow__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(unwrap_tensor_data(other) ** self.data)

    def __truediv__(self, other: Union[Tensor, float]) -> Tensor:
        return _call_function("_binary_op_nowarn", self, other, operator.truediv)

    def __rtruediv__(self, other: Union[Tensor, float]) -> Tensor:
        return _call_function("_binary_reverse_op_nowarn", self, other, operator.truediv)

    def __floordiv__(self, other: Union[Tensor, float]) -> Tensor:
        return _call_function("_binary_op_nowarn", self, other, operator.floordiv)

    def __rfloordiv__(self, other: Union[Tensor, float]) -> Tensor:
        return _call_function("_binary_reverse_op_nowarn", self, other, operator.floordiv)

    def __neg__(self) -> Tensor:
        return Tensor(-self.data)

    # Comparison operators

    def __lt__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data < unwrap_tensor_data(other))

    def __le__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data <= unwrap_tensor_data(other))

    def __eq__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data == unwrap_tensor_data(other))

    def __ne__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data != unwrap_tensor_data(other))

    def __gt__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data > unwrap_tensor_data(other))

    def __ge__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data >= unwrap_tensor_data(other))

    # Tensor functions

    def squeeze(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
        return _call_function("squeeze", self, axis)

    def flatten(self) -> Tensor:
        return _call_function("flatten", self)

    def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: Optional[bool] = False) -> Tensor:
        return _call_function("max", self, axis, keepdims)

    def min(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: Optional[bool] = False) -> Tensor:
        return _call_function("min", self, axis, keepdims)

    def abs(self) -> Tensor:
        return _call_function("abs", self)

    def isempty(self) -> bool:
        return _call_function("isempty", self)

    def astype(self, dtype: TensorDataType) -> Tensor:
        return _call_function("astype", self, dtype)

    def reshape(self, shape: Tuple[int, ...]) -> Tensor:
        return _call_function("reshape", self, shape)

    def item(self) -> float:
        return _call_function("item", self)

    def clone(self) -> float:
        return _call_function("clone", self)

