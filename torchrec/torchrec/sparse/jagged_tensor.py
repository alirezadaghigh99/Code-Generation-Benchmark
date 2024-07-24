def from_tensor_list(
        keys: List[str], tensors: List[torch.Tensor], key_dim: int = 1, cat_dim: int = 1
    ) -> "KeyedTensor":
        length_per_key = [tensor.shape[key_dim] for tensor in tensors]
        return KeyedTensor(
            keys=keys,
            length_per_key=length_per_key,
            values=torch.cat(tensors, dim=cat_dim),
            key_dim=key_dim,
        )

def from_offsets_sync(
        keys: List[str],
        values: torch.Tensor,
        offsets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> "KeyedJaggedTensor":
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            weights=weights,
            offsets=offsets,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            inverse_indices=inverse_indices,
        )
        return kjt.sync()

def from_lengths_sync(
        keys: List[str],
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> "KeyedJaggedTensor":
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            weights=weights,
            lengths=lengths,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            inverse_indices=inverse_indices,
        )
        return kjt.sync()

def empty(
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        values_dtype: Optional[torch.dtype] = None,
        weights_dtype: Optional[torch.dtype] = None,
        lengths_dtype: torch.dtype = torch.int32,
    ) -> "JaggedTensor":
        weights = (
            torch.empty(0, dtype=weights_dtype, device=device) if is_weighted else None
        )
        return JaggedTensor(
            values=torch.empty(0, dtype=values_dtype, device=device),
            offsets=torch.empty(0, dtype=lengths_dtype, device=device),
            lengths=torch.empty(0, dtype=lengths_dtype, device=device),
            weights=weights,
        )

def from_dense_lengths(
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> "JaggedTensor":
        """
        Constructs `JaggedTensor` from dense values/weights of shape (B, N,).

        Note that `lengths` is still of shape (B,).
        """

        mask2d = (
            _arange(end=values.size(1), device=values.device).expand(values.size(0), -1)
        ) < lengths.unsqueeze(-1)
        return JaggedTensor(
            values=values[mask2d],
            weights=_optional_mask(weights, mask2d),
            lengths=lengths,
        )

def from_dense(
        values: List[torch.Tensor],
        weights: Optional[List[torch.Tensor]] = None,
    ) -> "JaggedTensor":
        """
        Constructs `JaggedTensor` from dense values/weights of shape (B, N,).

        Note that `lengths` and `offsets` are still of shape (B,).

        Args:
            values (List[torch.Tensor]): a list of tensors for dense representation
            weights (Optional[List[torch.Tensor]]): if values have weights, tensor with
                the same shape as values.

        Returns:
            JaggedTensor: JaggedTensor created from 2D dense tensor.

        Example::

            values = [
                torch.Tensor([1.0]),
                torch.Tensor(),
                torch.Tensor([7.0, 8.0]),
                torch.Tensor([10.0, 11.0, 12.0]),
            ]
            weights = [
                torch.Tensor([1.0]),
                torch.Tensor(),
                torch.Tensor([7.0, 8.0]),
                torch.Tensor([10.0, 11.0, 12.0]),
            ]
            j1 = JaggedTensor.from_dense(
                values=values,
                weights=weights,
            )

            # j1 = [[1.0], [], [7.0], [8.0], [10.0, 11.0, 12.0]]
        """

        values_tensor = torch.cat(values, dim=0)
        lengths = torch.tensor(
            [value.size(0) for value in values],
            dtype=torch.int32,
            device=values_tensor.device,
        )
        weights_tensor = torch.cat(weights, dim=0) if weights is not None else None

        return JaggedTensor(
            values=values_tensor,
            weights=weights_tensor,
            lengths=lengths,
        )

def split(self, segments: List[int]) -> List["KeyedJaggedTensor"]:
        split_list: List[KeyedJaggedTensor] = []
        start = 0
        start_offset = 0
        _length_per_key = self.length_per_key()
        _offset_per_key = self.offset_per_key()
        for segment in segments:
            end = start + segment
            end_offset = _offset_per_key[end]
            keys: List[str] = self._keys[start:end]
            stride, stride_per_key_per_rank = (
                (None, self.stride_per_key_per_rank()[start:end])
                if self.variable_stride_per_key()
                else (self._stride, None)
            )
            if segment == len(self._keys):
                # no torch slicing required
                split_list.append(
                    KeyedJaggedTensor(
                        keys=self._keys,
                        values=self._values,
                        weights=self.weights_or_none(),
                        lengths=self._lengths,
                        offsets=self._offsets,
                        stride=stride,
                        stride_per_key_per_rank=stride_per_key_per_rank,
                        length_per_key=self._length_per_key,
                        offset_per_key=self._offset_per_key,
                        index_per_key=self._index_per_key,
                        jt_dict=self._jt_dict,
                        inverse_indices=None,
                    )
                )
            elif segment == 0:
                empty_int_list: List[int] = torch.jit.annotate(List[int], [])
                split_list.append(
                    KeyedJaggedTensor(
                        keys=keys,
                        values=torch.tensor(
                            empty_int_list,
                            device=self.device(),
                            dtype=self._values.dtype,
                        ),
                        weights=(
                            None
                            if self.weights_or_none() is None
                            else torch.tensor(
                                empty_int_list,
                                device=self.device(),
                                dtype=self.weights().dtype,
                            )
                        ),
                        lengths=torch.tensor(
                            empty_int_list, device=self.device(), dtype=torch.int
                        ),
                        offsets=torch.tensor(
                            empty_int_list, device=self.device(), dtype=torch.int
                        ),
                        stride=stride,
                        stride_per_key_per_rank=stride_per_key_per_rank,
                        length_per_key=None,
                        offset_per_key=None,
                        index_per_key=None,
                        jt_dict=None,
                        inverse_indices=None,
                    )
                )
            else:
                split_length_per_key = _length_per_key[start:end]

                if not torch.jit.is_scripting() and is_non_strict_exporting():
                    sz = sum(split_length_per_key)

                    [torch._check_is_size(length) for length in split_length_per_key]
                    torch._check(start_offset <= self._values.size(0))
                    torch._check(sz <= self._values.size(0))
                    torch._check_is_size(start_offset)

                    torch._check(start_offset + sz <= self._values.size(0))

                    lengths_start = self.lengths_offset_per_key()[start]
                    lengths_sz = self.lengths_offset_per_key()[end] - lengths_start

                    _lengths = torch.narrow(
                        self.lengths(), 0, lengths_start, lengths_sz
                    )

                    if self.weights_or_none() is not None:
                        torch._check(start_offset + sz <= self.weights().size(0))
                        torch._check(start_offset <= self.weights().size(0))

                    split_list.append(
                        KeyedJaggedTensor(
                            keys=keys,
                            values=torch.narrow(self._values, 0, start_offset, sz),
                            weights=(
                                None
                                if self.weights_or_none() is None
                                else torch.narrow(self.weights(), 0, start_offset, sz)
                            ),
                            lengths=_lengths,
                            offsets=None,
                            stride=stride,
                            stride_per_key_per_rank=stride_per_key_per_rank,
                            length_per_key=split_length_per_key,
                            offset_per_key=None,
                            index_per_key=None,
                            jt_dict=None,
                            inverse_indices=None,
                        )
                    )
                else:
                    pt2_checks_tensor_slice(self._values, start_offset, end_offset)

                    lengths_offset_per_key: List[int] = self.lengths_offset_per_key()
                    pt2_checks_tensor_slice(
                        self.lengths(),
                        lengths_offset_per_key[start],
                        lengths_offset_per_key[end],
                    )

                    split_list.append(
                        KeyedJaggedTensor(
                            keys=keys,
                            values=self._values[start_offset:end_offset],
                            weights=(
                                None
                                if self.weights_or_none() is None
                                else self.weights()[start_offset:end_offset]
                            ),
                            lengths=self.lengths()[
                                lengths_offset_per_key[start] : lengths_offset_per_key[
                                    end
                                ]
                            ],
                            offsets=None,
                            stride=stride,
                            stride_per_key_per_rank=stride_per_key_per_rank,
                            length_per_key=split_length_per_key,
                            offset_per_key=None,
                            index_per_key=None,
                            jt_dict=None,
                            inverse_indices=None,
                        )
                    )
            start = end
            start_offset = end_offset
        return split_list

class KeyedJaggedTensor(Pipelineable, metaclass=JaggedTensorMeta):
    """Represents an (optionally weighted) keyed jagged tensor.

    A `KeyedJaggedTensor` is a tensor with a *jagged dimension* which is dimension whose
    slices may be of different lengths. Keyed on first dimension and jagged on the last
    dimension.

    Implementation is torch.jit.script-able.

    Args:
        keys (List[str]): keys to the jagged Tensor.
        values (torch.Tensor): values tensor in dense representation.
        weights (Optional[torch.Tensor]): if the values have weights. Tensor with the
            same shape as values.
        lengths (Optional[torch.Tensor]): jagged slices, represented as lengths.
        offsets (Optional[torch.Tensor]): jagged slices, represented as cumulative
            offsets.
        stride (Optional[int]): number of examples per batch.
        stride_per_key_per_rank (Optional[List[List[int]]]): batch size
            (number of examples) per key per rank, with the outer list representing the
            keys and the inner list representing the values.
            Each value in the inner list represents the number of examples in the batch
            from the rank of its index in a distributed context.
        length_per_key (Optional[List[int]]): start length for each key.
        offset_per_key (Optional[List[int]]): start offset for each key and final
            offset.
        index_per_key (Optional[Dict[str, int]]): index for each key.
        jt_dict (Optional[Dict[str, JaggedTensor]]):
        inverse_indices (Optional[Tuple[List[str], torch.Tensor]]): inverse indices to
            expand deduplicated embedding output for variable stride per key.

    Example::

        #              0       1        2  <-- dim_1
        # "Feature0"   [V0,V1] None    [V2]
        # "Feature1"   [V3]    [V4]    [V5,V6,V7]
        #   ^
        #  dim_0

        dim_0: keyed dimension (ie. `Feature0`, `Feature1`)
        dim_1: optional second dimension (ie. batch size)
        dim_2: The jagged dimension which has slice lengths between 0-3 in the above example

        # We represent this data with following inputs:

        values: torch.Tensor = [V0, V1, V2, V3, V4, V5, V6, V7]  # V == any tensor datatype
        weights: torch.Tensor = [W0, W1, W2, W3, W4, W5, W6, W7]  # W == any tensor datatype
        lengths: torch.Tensor = [2, 0, 1, 1, 1, 3]  # representing the jagged slice
        offsets: torch.Tensor = [0, 2, 2, 3, 4, 5, 8]  # offsets from 0 for each jagged slice
        keys: List[str] = ["Feature0", "Feature1"]  # correspond to each value of dim_0
        index_per_key: Dict[str, int] = {"Feature0": 0, "Feature1": 1}  # index for each key
        offset_per_key: List[int] = [0, 3, 8]  # start offset for each key and final offset
    """

    # This is the subset of fields on KJT which are required (all other fields
    # can be derived from these fields, and are only cached)
    _fields = [
        "_values",
        "_weights",
        "_lengths",
        "_offsets",
    ]

    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        # Below exposed to ensure torch.script-able
        length_per_key: Optional[List[int]] = None,
        offset_per_key: Optional[List[int]] = None,
        index_per_key: Optional[Dict[str, int]] = None,
        jt_dict: Optional[Dict[str, JaggedTensor]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> None:
        self._keys: List[str] = keys
        self._values: torch.Tensor = values
        self._weights: Optional[torch.Tensor] = weights
        if offsets is not None:
            _assert_tensor_has_no_elements_or_has_integers(offsets, "offsets")
        if lengths is not None:
            _assert_tensor_has_no_elements_or_has_integers(lengths, "lengths")
        self._lengths: Optional[torch.Tensor] = lengths
        self._offsets: Optional[torch.Tensor] = offsets

        self._stride_per_key_per_rank: List[List[int]] = []
        self._stride_per_key: List[int] = []
        self._variable_stride_per_key: bool = False
        self._stride: int = -1

        if stride_per_key_per_rank is not None:
            if stride is not None:
                raise ValueError(
                    "Cannot initialize KJT with both `stride` and `stride_per_key_per_rank`"
                )
            self._stride_per_key_per_rank = stride_per_key_per_rank
            self._stride_per_key = [sum(s) for s in self._stride_per_key_per_rank]
            self._variable_stride_per_key = True
            if stride_per_key_per_rank is not None:
                self._stride = 0
            elif all(s == self.stride_per_key()[0] for s in self.stride_per_key()):
                self._stride = self.stride_per_key()[0]
        else:
            stride = _maybe_compute_stride_kjt(keys, stride, lengths, offsets)
            self._stride = stride
            self._stride_per_key_per_rank = [[stride]] * len(self._keys)
            self._stride_per_key = [sum(s) for s in self._stride_per_key_per_rank]

        # lazy fields
        self._length_per_key: Optional[List[int]] = length_per_key
        self._offset_per_key: Optional[List[int]] = offset_per_key
        self._index_per_key: Optional[Dict[str, int]] = index_per_key
        self._jt_dict: Optional[Dict[str, JaggedTensor]] = jt_dict
        self._inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = (
            inverse_indices
        )
        self._lengths_offset_per_key: List[int] = []

        self._init_pt2_checks()

    def _init_pt2_checks(self) -> None:
        if torch.jit.is_scripting() or not is_torchdynamo_compiling():
            return

        pt2_checks_all_is_size(self._stride_per_key)
        for s in self._stride_per_key_per_rank:
            pt2_checks_all_is_size(s)

    @staticmethod
    def from_offsets_sync(
        keys: List[str],
        values: torch.Tensor,
        offsets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> "KeyedJaggedTensor":
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            weights=weights,
            offsets=offsets,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            inverse_indices=inverse_indices,
        )
        return kjt.sync()

    @staticmethod
    def from_lengths_sync(
        keys: List[str],
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> "KeyedJaggedTensor":
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            weights=weights,
            lengths=lengths,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            inverse_indices=inverse_indices,
        )
        return kjt.sync()

    @staticmethod
    def concat(
        kjt_list: List["KeyedJaggedTensor"],
    ) -> "KeyedJaggedTensor":
        if len(kjt_list) == 0:
            raise ValueError("Can't concat empty KJT list")

        is_weighted: bool = kjt_list[0].weights_or_none() is not None
        has_length_per_key: bool = True

        length_per_key: List[int] = []
        keys: List[str] = []
        value_list: List[torch.Tensor] = []
        weight_list: List[torch.Tensor] = []
        length_list: List[torch.Tensor] = []
        stride_per_key_per_rank: List[List[int]] = []
        stride: Optional[int] = None
        variable_stride_per_key_list = [
            kjt.variable_stride_per_key() for kjt in kjt_list
        ]
        assert all(variable_stride_per_key_list) or not any(
            variable_stride_per_key_list
        ), "variable stride per key must be consistent for all KJTs"
        variable_stride_per_key = all(variable_stride_per_key_list)

        for kjt in kjt_list:
            curr_is_weighted: bool = kjt.weights_or_none() is not None
            if is_weighted != curr_is_weighted:
                raise ValueError("Can't merge weighted KJT with unweighted KJT")
            _length_per_key: Optional[List[int]] = None
            if kjt._length_per_key is None:
                has_length_per_key = False
            else:
                _length_per_key = kjt._length_per_key
            if has_length_per_key and _length_per_key is not None:
                length_per_key += _length_per_key
            keys += kjt.keys()
            value_list.append(kjt.values())
            if is_weighted:
                weight_list.append(kjt.weights())
            length_list.append(kjt.lengths())
            if variable_stride_per_key:
                stride_per_key_per_rank += kjt.stride_per_key_per_rank()
            elif stride is None:
                stride = kjt.stride()
            else:
                assert stride == kjt.stride(), "strides must be consistent for all KJTs"

        return KeyedJaggedTensor(
            keys=keys,
            values=torch.cat(value_list, dim=0),
            weights=torch.cat(weight_list, dim=0) if is_weighted else None,
            lengths=torch.cat(length_list, dim=0),
            stride=stride,
            stride_per_key_per_rank=(
                stride_per_key_per_rank if variable_stride_per_key else None
            ),
            length_per_key=length_per_key if has_length_per_key else None,
        )

    @staticmethod
    def empty(
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        values_dtype: Optional[torch.dtype] = None,
        weights_dtype: Optional[torch.dtype] = None,
        lengths_dtype: torch.dtype = torch.int32,
    ) -> "KeyedJaggedTensor":
        weights = (
            torch.empty(0, dtype=weights_dtype, device=device) if is_weighted else None
        )
        return KeyedJaggedTensor(
            keys=torch.jit.annotate(List[str], []),
            values=torch.empty(0, dtype=values_dtype, device=device),
            weights=weights,
            lengths=torch.empty(0, dtype=lengths_dtype, device=device),
            stride=0,
        )

    @staticmethod
    def empty_like(kjt: "KeyedJaggedTensor") -> "KeyedJaggedTensor":
        return _kjt_empty_like(kjt)

    @staticmethod
    def from_jt_dict(jt_dict: Dict[str, JaggedTensor]) -> "KeyedJaggedTensor":
        """
        Constructs a KeyedJaggedTensor from a Dict[str, JaggedTensor],
        but this function will ONLY work if the JaggedTensors all
        have the same "implicit" batch_size dimension.

        Basically, we can visualize JaggedTensors as 2-D tensors
        of the format of [batch_size x variable_feature_dim].
        In case, we have some batch without a feature value,
        the input JaggedTensor could just not include any values.

        But KeyedJaggedTensor (by default) typically pad "None"
        so that all the JaggedTensors stored in the KeyedJaggedTensor
        have the same batch_size dimension. That is, in the case,
        the JaggedTensor input didn't automatically pad
        for the empty batches, this function would error / not work.

        Consider the visualization of the following KeyedJaggedTensor:
        #              0       1        2  <-- dim_1
        # "Feature0"   [V0,V1] None    [V2]
        # "Feature1"   [V3]    [V4]    [V5,V6,V7]
        #   ^
        #  dim_0

        Notice that the inputs for this KeyedJaggedTensor would have looked like:
            values: torch.Tensor = [V0, V1, V2, V3, V4, V5, V6, V7]  # V == any tensor datatype
            weights: torch.Tensor = [W0, W1, W2, W3, W4, W5, W6, W7]  # W == any tensor datatype
            lengths: torch.Tensor = [2, 0, 1, 1, 1, 3]  # representing the jagged slice
            offsets: torch.Tensor = [0, 2, 2, 3, 4, 5, 8]  # offsets from 0 for each jagged slice
            keys: List[str] = ["Feature0", "Feature1"]  # correspond to each value of dim_0
            index_per_key: Dict[str, int] = {"Feature0": 0, "Feature1": 1}  # index for each key
            offset_per_key: List[int] = [0, 3, 8]  # start offset for each key and final offset

        Now if the input jt_dict = {
            # "Feature0"   [V0,V1] [V2]
            # "Feature1"   [V3]    [V4]    [V5,V6,V7]
        } and the "None" is left out from each JaggedTensor,
        then this function would fail as we would not correctly
        be able to pad "None" as it does not technically know
        the correct batch / place to pad within the JaggedTensor.

        Essentially, the lengths Tensor inferred by this function
        would be [2, 1, 1, 1, 3] indicating variable batch_size
        dim_1 violates the existing assumption / precondition
        that KeyedJaggedTensor's should have fixed batch_size dimension.

        """
        kjt_keys = list(jt_dict.keys())
        kjt_vals_list: List[torch.Tensor] = []
        kjt_lens_list: List[torch.Tensor] = []
        kjt_weights_list: List[torch.Tensor] = []
        stride_per_key: List[int] = []
        for jt in jt_dict.values():
            stride_per_key.append(len(jt.lengths()))
            kjt_vals_list.append(jt.values())
            kjt_lens_list.append(jt.lengths())
            weight = jt.weights_or_none()
            if weight is not None:
                kjt_weights_list.append(weight)
        kjt_vals = torch.concat(kjt_vals_list)
        kjt_lens = torch.concat(kjt_lens_list)
        kjt_weights = (
            torch.concat(kjt_weights_list) if len(kjt_weights_list) > 0 else None
        )
        kjt_stride, kjt_stride_per_key_per_rank = (
            (stride_per_key[0], None)
            if all(s == stride_per_key[0] for s in stride_per_key)
            else (None, [[stride] for stride in stride_per_key])
        )
        kjt = KeyedJaggedTensor(
            keys=kjt_keys,
            values=kjt_vals,
            weights=kjt_weights,
            lengths=kjt_lens,
            stride=kjt_stride,
            stride_per_key_per_rank=kjt_stride_per_key_per_rank,
        ).sync()
        return kjt

    def sync(self) -> "KeyedJaggedTensor":
        if not is_torchdynamo_compiling():
            self.length_per_key()
            self.offset_per_key()
        return self

    def unsync(self) -> "KeyedJaggedTensor":
        self._length_per_key = None
        self._offset_per_key = None
        return self

    def device(self) -> torch.device:
        return self._values.device

    def lengths(self) -> torch.Tensor:
        _lengths = _maybe_compute_lengths(self._lengths, self._offsets)
        self._lengths = _lengths
        return _lengths

    def lengths_or_none(self) -> Optional[torch.Tensor]:
        return self._lengths

    def offsets(self) -> torch.Tensor:
        _offsets = _maybe_compute_offsets(self._lengths, self._offsets)
        self._offsets = _offsets
        return _offsets

    def offsets_or_none(self) -> Optional[torch.Tensor]:
        return self._offsets

    def keys(self) -> List[str]:
        return self._keys

    def values(self) -> torch.Tensor:
        return self._values

    def weights(self) -> torch.Tensor:
        return _get_weights_or_throw(self._weights)

    def weights_or_none(self) -> Optional[torch.Tensor]:
        return self._weights

    def stride(self) -> int:
        return self._stride

    def stride_per_key(self) -> List[int]:
        return self._stride_per_key

    def stride_per_key_per_rank(self) -> List[List[int]]:
        return self._stride_per_key_per_rank

    def variable_stride_per_key(self) -> bool:
        return self._variable_stride_per_key

    def inverse_indices(self) -> Tuple[List[str], torch.Tensor]:
        return _get_inverse_indices_or_throw(self._inverse_indices)

    def inverse_indices_or_none(self) -> Optional[Tuple[List[str], torch.Tensor]]:
        return self._inverse_indices

    def _key_indices(self) -> Dict[str, int]:
        _index_per_key: Dict[str, int] = _maybe_compute_index_per_key(
            self._keys,
            self._index_per_key,
        )
        self._index_per_key = _index_per_key
        return _index_per_key

    def length_per_key(self) -> List[int]:
        _length_per_key = _maybe_compute_length_per_key(
            keys=self._keys,
            stride=self.stride(),
            stride_per_key=self.stride_per_key(),
            variable_stride_per_key=self.variable_stride_per_key(),
            length_per_key=self._length_per_key,
            lengths=self._lengths,
            offsets=self._offsets,
            values=self._values,
        )
        self._length_per_key = _length_per_key
        return _length_per_key

    def length_per_key_or_none(self) -> Optional[List[int]]:
        return self._length_per_key

    def offset_per_key(self) -> List[int]:
        _length_per_key, _offset_per_key = _maybe_compute_offset_per_key(
            keys=self._keys,
            stride=self.stride(),
            stride_per_key=self.stride_per_key(),
            variable_stride_per_key=self.variable_stride_per_key(),
            length_per_key=self._length_per_key,
            offset_per_key=self._offset_per_key,
            lengths=self._lengths,
            offsets=self._offsets,
            values=self._values,
        )
        self._length_per_key = _length_per_key
        self._offset_per_key = _offset_per_key
        return _offset_per_key

    def offset_per_key_or_none(self) -> Optional[List[int]]:
        return self._offset_per_key

    def lengths_offset_per_key(self) -> List[int]:
        if not self._lengths_offset_per_key:
            self._lengths_offset_per_key = _cumsum(self.stride_per_key())
        return self._lengths_offset_per_key

    def index_per_key(self) -> Dict[str, int]:
        return self._key_indices()

    def split(self, segments: List[int]) -> List["KeyedJaggedTensor"]:
        split_list: List[KeyedJaggedTensor] = []
        start = 0
        start_offset = 0
        _length_per_key = self.length_per_key()
        _offset_per_key = self.offset_per_key()
        for segment in segments:
            end = start + segment
            end_offset = _offset_per_key[end]
            keys: List[str] = self._keys[start:end]
            stride, stride_per_key_per_rank = (
                (None, self.stride_per_key_per_rank()[start:end])
                if self.variable_stride_per_key()
                else (self._stride, None)
            )
            if segment == len(self._keys):
                # no torch slicing required
                split_list.append(
                    KeyedJaggedTensor(
                        keys=self._keys,
                        values=self._values,
                        weights=self.weights_or_none(),
                        lengths=self._lengths,
                        offsets=self._offsets,
                        stride=stride,
                        stride_per_key_per_rank=stride_per_key_per_rank,
                        length_per_key=self._length_per_key,
                        offset_per_key=self._offset_per_key,
                        index_per_key=self._index_per_key,
                        jt_dict=self._jt_dict,
                        inverse_indices=None,
                    )
                )
            elif segment == 0:
                empty_int_list: List[int] = torch.jit.annotate(List[int], [])
                split_list.append(
                    KeyedJaggedTensor(
                        keys=keys,
                        values=torch.tensor(
                            empty_int_list,
                            device=self.device(),
                            dtype=self._values.dtype,
                        ),
                        weights=(
                            None
                            if self.weights_or_none() is None
                            else torch.tensor(
                                empty_int_list,
                                device=self.device(),
                                dtype=self.weights().dtype,
                            )
                        ),
                        lengths=torch.tensor(
                            empty_int_list, device=self.device(), dtype=torch.int
                        ),
                        offsets=torch.tensor(
                            empty_int_list, device=self.device(), dtype=torch.int
                        ),
                        stride=stride,
                        stride_per_key_per_rank=stride_per_key_per_rank,
                        length_per_key=None,
                        offset_per_key=None,
                        index_per_key=None,
                        jt_dict=None,
                        inverse_indices=None,
                    )
                )
            else:
                split_length_per_key = _length_per_key[start:end]

                if not torch.jit.is_scripting() and is_non_strict_exporting():
                    sz = sum(split_length_per_key)

                    [torch._check_is_size(length) for length in split_length_per_key]
                    torch._check(start_offset <= self._values.size(0))
                    torch._check(sz <= self._values.size(0))
                    torch._check_is_size(start_offset)

                    torch._check(start_offset + sz <= self._values.size(0))

                    lengths_start = self.lengths_offset_per_key()[start]
                    lengths_sz = self.lengths_offset_per_key()[end] - lengths_start

                    _lengths = torch.narrow(
                        self.lengths(), 0, lengths_start, lengths_sz
                    )

                    if self.weights_or_none() is not None:
                        torch._check(start_offset + sz <= self.weights().size(0))
                        torch._check(start_offset <= self.weights().size(0))

                    split_list.append(
                        KeyedJaggedTensor(
                            keys=keys,
                            values=torch.narrow(self._values, 0, start_offset, sz),
                            weights=(
                                None
                                if self.weights_or_none() is None
                                else torch.narrow(self.weights(), 0, start_offset, sz)
                            ),
                            lengths=_lengths,
                            offsets=None,
                            stride=stride,
                            stride_per_key_per_rank=stride_per_key_per_rank,
                            length_per_key=split_length_per_key,
                            offset_per_key=None,
                            index_per_key=None,
                            jt_dict=None,
                            inverse_indices=None,
                        )
                    )
                else:
                    pt2_checks_tensor_slice(self._values, start_offset, end_offset)

                    lengths_offset_per_key: List[int] = self.lengths_offset_per_key()
                    pt2_checks_tensor_slice(
                        self.lengths(),
                        lengths_offset_per_key[start],
                        lengths_offset_per_key[end],
                    )

                    split_list.append(
                        KeyedJaggedTensor(
                            keys=keys,
                            values=self._values[start_offset:end_offset],
                            weights=(
                                None
                                if self.weights_or_none() is None
                                else self.weights()[start_offset:end_offset]
                            ),
                            lengths=self.lengths()[
                                lengths_offset_per_key[start] : lengths_offset_per_key[
                                    end
                                ]
                            ],
                            offsets=None,
                            stride=stride,
                            stride_per_key_per_rank=stride_per_key_per_rank,
                            length_per_key=split_length_per_key,
                            offset_per_key=None,
                            index_per_key=None,
                            jt_dict=None,
                            inverse_indices=None,
                        )
                    )
            start = end
            start_offset = end_offset
        return split_list

    def permute(
        self, indices: List[int], indices_tensor: Optional[torch.Tensor] = None
    ) -> "KeyedJaggedTensor":

        if indices_tensor is None:
            indices_tensor = torch.tensor(
                indices, dtype=torch.int, device=self.device()
            )

        length_per_key = self.length_per_key()
        permuted_keys: List[str] = []
        permuted_stride_per_key_per_rank: List[List[int]] = []
        permuted_length_per_key: List[int] = []
        permuted_length_per_key_sum = 0
        for index in indices:
            key = self.keys()[index]
            permuted_keys.append(key)
            permuted_stride_per_key_per_rank.append(
                self.stride_per_key_per_rank()[index]
            )
            permuted_length_per_key.append(length_per_key[index])

        permuted_length_per_key_sum = sum(permuted_length_per_key)
        if not torch.jit.is_scripting() and is_non_strict_exporting():
            torch._check_is_size(permuted_length_per_key_sum)
            torch._check(permuted_length_per_key_sum != -1)
            torch._check(permuted_length_per_key_sum != 0)

        if self.variable_stride_per_key():
            length_per_key_tensor = _pin_and_move(
                torch.tensor(self.length_per_key()), self.device()
            )
            stride_per_key_tensor = _pin_and_move(
                torch.tensor(self.stride_per_key()), self.device()
            )
            permuted_lengths, _ = _permute_tensor_by_segments(
                self.lengths(),
                stride_per_key_tensor,
                indices_tensor,
                None,
            )
            permuted_values, permuted_weights = _permute_tensor_by_segments(
                self.values(),
                length_per_key_tensor,
                indices_tensor,
                self.weights_or_none(),
            )
        else:

            (
                permuted_lengths,
                permuted_values,
                permuted_weights,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                indices_tensor,
                self.lengths().view(len(self._keys), -1),
                self.values(),
                self.weights_or_none(),
                permuted_length_per_key_sum,
            )
        stride, optional_permuted_stride_per_key_per_rank = (
            (None, permuted_stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )
        kjt = KeyedJaggedTensor(
            keys=permuted_keys,
            values=permuted_values,
            weights=permuted_weights,
            lengths=permuted_lengths.view(-1),
            offsets=None,
            stride=stride,
            stride_per_key_per_rank=optional_permuted_stride_per_key_per_rank,
            length_per_key=permuted_length_per_key if len(permuted_keys) > 0 else None,
            offset_per_key=None,
            index_per_key=None,
            jt_dict=None,
            inverse_indices=None,
        )
        return kjt

    def flatten_lengths(self) -> "KeyedJaggedTensor":
        stride, stride_per_key_per_rank = (
            (None, self.stride_per_key_per_rank())
            if self.variable_stride_per_key()
            else (self._stride, None)
        )
        return KeyedJaggedTensor(
            keys=self._keys,
            values=self._values,
            weights=self._weights,
            lengths=self.lengths().view(-1),
            offsets=None,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            length_per_key=self.length_per_key(),
            offset_per_key=None,
            index_per_key=None,
            jt_dict=None,
            inverse_indices=None,
        )

    def __getitem__(self, key: str) -> JaggedTensor:
        offset_per_key = self.offset_per_key()
        index = self._key_indices()[key]
        start_offset = offset_per_key[index]
        end_offset = (
            offset_per_key[index + 1]
            if index + 1 < len(offset_per_key)
            else start_offset
        )

        if not torch.jit.is_scripting() and is_non_strict_exporting():
            length_per_key = self.length_per_key()
            _lengths = torch.narrow(
                self.lengths(),
                0,
                self.lengths_offset_per_key()[index],
                self.lengths_offset_per_key()[index + 1]
                - self.lengths_offset_per_key()[index],
            )
            sz = length_per_key[index]

            torch._check_is_size(start_offset)
            torch._check_is_size(sz)
            torch._check(start_offset <= self.values().size(0))
            torch._check(sz <= self.values().size(0))

            if self.weights_or_none() is not None:
                torch._check(start_offset <= self.weights().size(0))
                torch._check(sz <= self.weights().size(0))

            return JaggedTensor(
                values=torch.narrow(
                    self.values(),
                    0,
                    start_offset,
                    sz,
                ),
                weights=(
                    None
                    if self.weights_or_none() is None
                    else torch.narrow(
                        self.weights(),
                        0,
                        start_offset,
                        sz,
                    )
                ),
                lengths=_lengths,
                offsets=None,
            )
        else:
            pt2_checks_tensor_slice(self._values, start_offset, end_offset)

            return JaggedTensor(
                values=self._values[start_offset:end_offset],
                weights=(
                    None
                    if self.weights_or_none() is None
                    else self.weights()[start_offset:end_offset]
                ),
                lengths=self.lengths()[
                    self.lengths_offset_per_key()[
                        index
                    ] : self.lengths_offset_per_key()[index + 1]
                ],
                offsets=None,
            )

    def to_dict(self) -> Dict[str, JaggedTensor]:
        _jt_dict = _maybe_compute_kjt_to_jt_dict(
            stride=self.stride(),
            stride_per_key=self.stride_per_key(),
            keys=self.keys(),
            length_per_key=self.length_per_key(),
            lengths=self.lengths(),
            values=self.values(),
            variable_stride_per_key=self.variable_stride_per_key(),
            weights=self.weights_or_none(),
            jt_dict=self._jt_dict,
        )
        self._jt_dict = _jt_dict
        return _jt_dict

    @torch.jit.unused
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        self._values.record_stream(stream)
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        if weights is not None:
            weights.record_stream(stream)
        if lengths is not None:
            lengths.record_stream(stream)
        if offsets is not None:
            offsets.record_stream(stream)

    def to(
        self,
        device: torch.device,
        non_blocking: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "KeyedJaggedTensor":
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride, stride_per_key_per_rank = (
            (None, self._stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )
        length_per_key = self._length_per_key
        offset_per_key = self._offset_per_key
        index_per_key = self._index_per_key
        jt_dict = self._jt_dict
        inverse_indices = self._inverse_indices
        if inverse_indices is not None:
            inverse_indices = (
                inverse_indices[0],
                inverse_indices[1].to(device, non_blocking=non_blocking),
            )
        if weights is not None:
            if dtype is not None:
                weights = weights.to(
                    dtype=dtype, device=device, non_blocking=non_blocking
                )
            else:
                weights = weights.to(device=device, non_blocking=non_blocking)

        return KeyedJaggedTensor(
            keys=self._keys,
            values=self._values.to(device, non_blocking=non_blocking),
            weights=weights,
            lengths=(
                lengths.to(device, non_blocking=non_blocking)
                if lengths is not None
                else None
            ),
            offsets=(
                offsets.to(device, non_blocking=non_blocking)
                if offsets is not None
                else None
            ),
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            length_per_key=length_per_key,
            offset_per_key=offset_per_key,
            index_per_key=index_per_key,
            jt_dict=jt_dict,
            inverse_indices=inverse_indices,
        )

    def __str__(self) -> str:
        if len(self._keys) == 0 or self._offsets is None and self._lengths is None:
            return "KeyedJaggedTensor()\n"
        offsets = self.offsets()

        return (
            "KeyedJaggedTensor({\n"
            + ",\n".join(
                [
                    "    "
                    + _jagged_tensor_string(
                        self._keys[index],
                        self._values,
                        self._weights,
                        offsets,
                        sum(self.stride_per_key()[:index]),
                        sum(self.stride_per_key()[: index + 1]),
                    )
                    for index in range(len(self._keys))
                ]
            )
            + "\n})\n"
        )

    def pin_memory(self) -> "KeyedJaggedTensor":
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride, stride_per_key_per_rank = (
            (None, self._stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )
        inverse_indices = self._inverse_indices
        if inverse_indices is not None:
            inverse_indices = (inverse_indices[0], inverse_indices[1].pin_memory())

        return KeyedJaggedTensor(
            keys=self._keys,
            values=self._values.pin_memory(),
            weights=weights.pin_memory() if weights is not None else None,
            lengths=lengths.pin_memory() if lengths is not None else None,
            offsets=offsets.pin_memory() if offsets is not None else None,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            length_per_key=self._length_per_key,
            offset_per_key=self._offset_per_key,
            index_per_key=self._index_per_key,
            jt_dict=None,
            inverse_indices=inverse_indices,
        )

    def dist_labels(self) -> List[str]:
        labels = ["lengths", "values"]
        if self.variable_stride_per_key():
            labels.append("strides")
        if self.weights_or_none() is not None:
            labels.append("weights")
        return labels

    def dist_splits(self, key_splits: List[int]) -> List[List[int]]:
        batch_size_per_split = _sum_by_splits(self.stride_per_key(), key_splits)
        length_per_split = _sum_by_splits(self.length_per_key(), key_splits)
        splits = [batch_size_per_split, length_per_split]
        if self.variable_stride_per_key():
            splits.append(key_splits)
        if self.weights_or_none() is not None:
            splits.append(length_per_split)
        return splits

    def dist_tensors(self) -> List[torch.Tensor]:
        tensors = [self.lengths(), self.values()]
        if self.variable_stride_per_key():
            strides = _pin_and_move(torch.tensor(self.stride_per_key()), self.device())
            tensors.append(strides)
        if self.weights_or_none() is not None:
            tensors.append(self.weights())
        return tensors

    @staticmethod
    def dist_init(
        keys: List[str],
        tensors: List[torch.Tensor],
        variable_stride_per_key: bool,
        num_workers: int,
        recat: Optional[torch.Tensor],
        stride_per_rank: Optional[List[int]],
        stagger: int = 1,
    ) -> "KeyedJaggedTensor":
        assert len(tensors) in [2, 3, 4]
        lengths = tensors[0]
        values = tensors[1]
        stride_per_rank_per_key = tensors[2] if variable_stride_per_key else None
        weights = (
            tensors[-1]
            if (variable_stride_per_key and len(tensors) == 4)
            or (not variable_stride_per_key and len(tensors) == 3)
            else None
        )

        if variable_stride_per_key:
            assert stride_per_rank_per_key is not None
            stride_per_key_per_rank_tensor: torch.Tensor = stride_per_rank_per_key.view(
                num_workers, len(keys)
            ).T.cpu()

            strides_cumsum: torch.Tensor = (
                torch.ops.fbgemm.asynchronous_complete_cumsum(stride_per_rank_per_key)
            ).cpu()

            cumsum_lengths = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

            n = strides_cumsum.size(0)
            strides_cumsum_from_1 = torch.narrow(
                strides_cumsum, dim=0, start=1, length=n - 1
            )
            strides_cumsum_to_minus_1 = torch.narrow(
                strides_cumsum, dim=0, start=0, length=n - 1
            )
            length_per_key_tensor = (
                cumsum_lengths[strides_cumsum_from_1]
                - cumsum_lengths[strides_cumsum_to_minus_1]
            )

            with record_function("## all2all_data:recat_values ##"):
                if recat is not None:
                    lengths, _ = _permute_tensor_by_segments(
                        lengths,
                        stride_per_rank_per_key,
                        torch.jit._unwrap_optional(recat),
                        None,
                    )
                    values, weights = _permute_tensor_by_segments(
                        values,
                        length_per_key_tensor,
                        torch.jit._unwrap_optional(recat),
                        weights,
                    )

            stride_per_key_per_rank = torch.jit.annotate(
                List[List[int]], stride_per_key_per_rank_tensor.tolist()
            )

            if not stride_per_key_per_rank:
                stride_per_key_per_rank = [[0]] * len(keys)
            if stagger > 1:
                stride_per_key_per_rank_stagger: List[List[int]] = []
                local_world_size = num_workers // stagger
                for i in range(len(keys)):
                    stride_per_rank_stagger: List[int] = []
                    for j in range(local_world_size):
                        stride_per_rank_stagger.extend(
                            stride_per_key_per_rank[i][j::local_world_size]
                        )
                    stride_per_key_per_rank_stagger.append(stride_per_rank_stagger)
                stride_per_key_per_rank = stride_per_key_per_rank_stagger

            kjt = KeyedJaggedTensor(
                keys=keys,
                values=values,
                weights=weights,
                lengths=lengths,
                stride_per_key_per_rank=stride_per_key_per_rank,
            )
            return kjt.sync()
        else:
            assert stride_per_rank is not None
            with record_function("## all2all_data:recat_values ##"):
                if recat is not None:
                    stride = stride_per_rank[0]

                    single_batch_per_rank = True
                    if not is_torchdynamo_compiling():
                        single_batch_per_rank = all(
                            s == stride for s in stride_per_rank
                        )

                    if single_batch_per_rank:
                        (
                            lengths,
                            values,
                            weights,
                        ) = torch.ops.fbgemm.permute_2D_sparse_data(
                            torch.jit._unwrap_optional(recat),
                            lengths.view(-1, stride),
                            values,
                            weights,
                            values.numel(),
                        )
                        lengths = lengths.view(-1)
                    else:  # variable batch size per rank
                        (
                            lengths,
                            values,
                            weights,
                        ) = torch.ops.fbgemm.permute_1D_sparse_data(
                            torch.jit._unwrap_optional(recat),
                            lengths.view(-1),
                            values,
                            weights,
                            values.numel(),
                        )
            kjt = KeyedJaggedTensor(
                keys=keys,
                values=values,
                weights=weights,
                lengths=lengths,
                stride=sum(stride_per_rank),
            )
            return kjt.sync()

class ComputeKJTToJTDict(torch.nn.Module):
    """Converts a KeyedJaggedTensor to a dict of JaggedTensors.

    Args:

    Example::
        #              0       1        2  <-- dim_1
        # "Feature0"   [V0,V1] None    [V2]
        # "Feature1"   [V3]    [V4]    [V5,V6,V7]
        #   ^
        #  dim_0

        would return

        {
            "Feature0": JaggedTensor([[V0,V1],None,V2]),
            "Feature1": JaggedTensor([V3,V4,[V5,V6,V7]]),
        }
    """

    def forward(
        self, keyed_jagged_tensor: "KeyedJaggedTensor"
    ) -> Dict[str, JaggedTensor]:
        """
        Converts a KeyedJaggedTensor into a dict of JaggedTensors.

        Args:
            keyed_jagged_tensor (KeyedJaggedTensor): tensor to convert
        Returns:
            Dict[str, JaggedTensor]
        """
        return _maybe_compute_kjt_to_jt_dict(
            stride=keyed_jagged_tensor.stride(),
            stride_per_key=keyed_jagged_tensor.stride_per_key(),
            keys=keyed_jagged_tensor.keys(),
            length_per_key=keyed_jagged_tensor.length_per_key(),
            values=keyed_jagged_tensor.values(),
            lengths=keyed_jagged_tensor.lengths(),
            variable_stride_per_key=keyed_jagged_tensor.variable_stride_per_key(),
            weights=keyed_jagged_tensor.weights_or_none(),
            jt_dict=keyed_jagged_tensor._jt_dict,
        )

class KeyedJaggedTensor(Pipelineable, metaclass=JaggedTensorMeta):
    """Represents an (optionally weighted) keyed jagged tensor.

    A `KeyedJaggedTensor` is a tensor with a *jagged dimension* which is dimension whose
    slices may be of different lengths. Keyed on first dimension and jagged on the last
    dimension.

    Implementation is torch.jit.script-able.

    Args:
        keys (List[str]): keys to the jagged Tensor.
        values (torch.Tensor): values tensor in dense representation.
        weights (Optional[torch.Tensor]): if the values have weights. Tensor with the
            same shape as values.
        lengths (Optional[torch.Tensor]): jagged slices, represented as lengths.
        offsets (Optional[torch.Tensor]): jagged slices, represented as cumulative
            offsets.
        stride (Optional[int]): number of examples per batch.
        stride_per_key_per_rank (Optional[List[List[int]]]): batch size
            (number of examples) per key per rank, with the outer list representing the
            keys and the inner list representing the values.
            Each value in the inner list represents the number of examples in the batch
            from the rank of its index in a distributed context.
        length_per_key (Optional[List[int]]): start length for each key.
        offset_per_key (Optional[List[int]]): start offset for each key and final
            offset.
        index_per_key (Optional[Dict[str, int]]): index for each key.
        jt_dict (Optional[Dict[str, JaggedTensor]]):
        inverse_indices (Optional[Tuple[List[str], torch.Tensor]]): inverse indices to
            expand deduplicated embedding output for variable stride per key.

    Example::

        #              0       1        2  <-- dim_1
        # "Feature0"   [V0,V1] None    [V2]
        # "Feature1"   [V3]    [V4]    [V5,V6,V7]
        #   ^
        #  dim_0

        dim_0: keyed dimension (ie. `Feature0`, `Feature1`)
        dim_1: optional second dimension (ie. batch size)
        dim_2: The jagged dimension which has slice lengths between 0-3 in the above example

        # We represent this data with following inputs:

        values: torch.Tensor = [V0, V1, V2, V3, V4, V5, V6, V7]  # V == any tensor datatype
        weights: torch.Tensor = [W0, W1, W2, W3, W4, W5, W6, W7]  # W == any tensor datatype
        lengths: torch.Tensor = [2, 0, 1, 1, 1, 3]  # representing the jagged slice
        offsets: torch.Tensor = [0, 2, 2, 3, 4, 5, 8]  # offsets from 0 for each jagged slice
        keys: List[str] = ["Feature0", "Feature1"]  # correspond to each value of dim_0
        index_per_key: Dict[str, int] = {"Feature0": 0, "Feature1": 1}  # index for each key
        offset_per_key: List[int] = [0, 3, 8]  # start offset for each key and final offset
    """

    # This is the subset of fields on KJT which are required (all other fields
    # can be derived from these fields, and are only cached)
    _fields = [
        "_values",
        "_weights",
        "_lengths",
        "_offsets",
    ]

    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        # Below exposed to ensure torch.script-able
        length_per_key: Optional[List[int]] = None,
        offset_per_key: Optional[List[int]] = None,
        index_per_key: Optional[Dict[str, int]] = None,
        jt_dict: Optional[Dict[str, JaggedTensor]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> None:
        self._keys: List[str] = keys
        self._values: torch.Tensor = values
        self._weights: Optional[torch.Tensor] = weights
        if offsets is not None:
            _assert_tensor_has_no_elements_or_has_integers(offsets, "offsets")
        if lengths is not None:
            _assert_tensor_has_no_elements_or_has_integers(lengths, "lengths")
        self._lengths: Optional[torch.Tensor] = lengths
        self._offsets: Optional[torch.Tensor] = offsets

        self._stride_per_key_per_rank: List[List[int]] = []
        self._stride_per_key: List[int] = []
        self._variable_stride_per_key: bool = False
        self._stride: int = -1

        if stride_per_key_per_rank is not None:
            if stride is not None:
                raise ValueError(
                    "Cannot initialize KJT with both `stride` and `stride_per_key_per_rank`"
                )
            self._stride_per_key_per_rank = stride_per_key_per_rank
            self._stride_per_key = [sum(s) for s in self._stride_per_key_per_rank]
            self._variable_stride_per_key = True
            if stride_per_key_per_rank is not None:
                self._stride = 0
            elif all(s == self.stride_per_key()[0] for s in self.stride_per_key()):
                self._stride = self.stride_per_key()[0]
        else:
            stride = _maybe_compute_stride_kjt(keys, stride, lengths, offsets)
            self._stride = stride
            self._stride_per_key_per_rank = [[stride]] * len(self._keys)
            self._stride_per_key = [sum(s) for s in self._stride_per_key_per_rank]

        # lazy fields
        self._length_per_key: Optional[List[int]] = length_per_key
        self._offset_per_key: Optional[List[int]] = offset_per_key
        self._index_per_key: Optional[Dict[str, int]] = index_per_key
        self._jt_dict: Optional[Dict[str, JaggedTensor]] = jt_dict
        self._inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = (
            inverse_indices
        )
        self._lengths_offset_per_key: List[int] = []

        self._init_pt2_checks()

    def _init_pt2_checks(self) -> None:
        if torch.jit.is_scripting() or not is_torchdynamo_compiling():
            return

        pt2_checks_all_is_size(self._stride_per_key)
        for s in self._stride_per_key_per_rank:
            pt2_checks_all_is_size(s)

    @staticmethod
    def from_offsets_sync(
        keys: List[str],
        values: torch.Tensor,
        offsets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> "KeyedJaggedTensor":
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            weights=weights,
            offsets=offsets,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            inverse_indices=inverse_indices,
        )
        return kjt.sync()

    @staticmethod
    def from_lengths_sync(
        keys: List[str],
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> "KeyedJaggedTensor":
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            weights=weights,
            lengths=lengths,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            inverse_indices=inverse_indices,
        )
        return kjt.sync()

    @staticmethod
    def concat(
        kjt_list: List["KeyedJaggedTensor"],
    ) -> "KeyedJaggedTensor":
        if len(kjt_list) == 0:
            raise ValueError("Can't concat empty KJT list")

        is_weighted: bool = kjt_list[0].weights_or_none() is not None
        has_length_per_key: bool = True

        length_per_key: List[int] = []
        keys: List[str] = []
        value_list: List[torch.Tensor] = []
        weight_list: List[torch.Tensor] = []
        length_list: List[torch.Tensor] = []
        stride_per_key_per_rank: List[List[int]] = []
        stride: Optional[int] = None
        variable_stride_per_key_list = [
            kjt.variable_stride_per_key() for kjt in kjt_list
        ]
        assert all(variable_stride_per_key_list) or not any(
            variable_stride_per_key_list
        ), "variable stride per key must be consistent for all KJTs"
        variable_stride_per_key = all(variable_stride_per_key_list)

        for kjt in kjt_list:
            curr_is_weighted: bool = kjt.weights_or_none() is not None
            if is_weighted != curr_is_weighted:
                raise ValueError("Can't merge weighted KJT with unweighted KJT")
            _length_per_key: Optional[List[int]] = None
            if kjt._length_per_key is None:
                has_length_per_key = False
            else:
                _length_per_key = kjt._length_per_key
            if has_length_per_key and _length_per_key is not None:
                length_per_key += _length_per_key
            keys += kjt.keys()
            value_list.append(kjt.values())
            if is_weighted:
                weight_list.append(kjt.weights())
            length_list.append(kjt.lengths())
            if variable_stride_per_key:
                stride_per_key_per_rank += kjt.stride_per_key_per_rank()
            elif stride is None:
                stride = kjt.stride()
            else:
                assert stride == kjt.stride(), "strides must be consistent for all KJTs"

        return KeyedJaggedTensor(
            keys=keys,
            values=torch.cat(value_list, dim=0),
            weights=torch.cat(weight_list, dim=0) if is_weighted else None,
            lengths=torch.cat(length_list, dim=0),
            stride=stride,
            stride_per_key_per_rank=(
                stride_per_key_per_rank if variable_stride_per_key else None
            ),
            length_per_key=length_per_key if has_length_per_key else None,
        )

    @staticmethod
    def empty(
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        values_dtype: Optional[torch.dtype] = None,
        weights_dtype: Optional[torch.dtype] = None,
        lengths_dtype: torch.dtype = torch.int32,
    ) -> "KeyedJaggedTensor":
        weights = (
            torch.empty(0, dtype=weights_dtype, device=device) if is_weighted else None
        )
        return KeyedJaggedTensor(
            keys=torch.jit.annotate(List[str], []),
            values=torch.empty(0, dtype=values_dtype, device=device),
            weights=weights,
            lengths=torch.empty(0, dtype=lengths_dtype, device=device),
            stride=0,
        )

    @staticmethod
    def empty_like(kjt: "KeyedJaggedTensor") -> "KeyedJaggedTensor":
        return _kjt_empty_like(kjt)

    @staticmethod
    def from_jt_dict(jt_dict: Dict[str, JaggedTensor]) -> "KeyedJaggedTensor":
        """
        Constructs a KeyedJaggedTensor from a Dict[str, JaggedTensor],
        but this function will ONLY work if the JaggedTensors all
        have the same "implicit" batch_size dimension.

        Basically, we can visualize JaggedTensors as 2-D tensors
        of the format of [batch_size x variable_feature_dim].
        In case, we have some batch without a feature value,
        the input JaggedTensor could just not include any values.

        But KeyedJaggedTensor (by default) typically pad "None"
        so that all the JaggedTensors stored in the KeyedJaggedTensor
        have the same batch_size dimension. That is, in the case,
        the JaggedTensor input didn't automatically pad
        for the empty batches, this function would error / not work.

        Consider the visualization of the following KeyedJaggedTensor:
        #              0       1        2  <-- dim_1
        # "Feature0"   [V0,V1] None    [V2]
        # "Feature1"   [V3]    [V4]    [V5,V6,V7]
        #   ^
        #  dim_0

        Notice that the inputs for this KeyedJaggedTensor would have looked like:
            values: torch.Tensor = [V0, V1, V2, V3, V4, V5, V6, V7]  # V == any tensor datatype
            weights: torch.Tensor = [W0, W1, W2, W3, W4, W5, W6, W7]  # W == any tensor datatype
            lengths: torch.Tensor = [2, 0, 1, 1, 1, 3]  # representing the jagged slice
            offsets: torch.Tensor = [0, 2, 2, 3, 4, 5, 8]  # offsets from 0 for each jagged slice
            keys: List[str] = ["Feature0", "Feature1"]  # correspond to each value of dim_0
            index_per_key: Dict[str, int] = {"Feature0": 0, "Feature1": 1}  # index for each key
            offset_per_key: List[int] = [0, 3, 8]  # start offset for each key and final offset

        Now if the input jt_dict = {
            # "Feature0"   [V0,V1] [V2]
            # "Feature1"   [V3]    [V4]    [V5,V6,V7]
        } and the "None" is left out from each JaggedTensor,
        then this function would fail as we would not correctly
        be able to pad "None" as it does not technically know
        the correct batch / place to pad within the JaggedTensor.

        Essentially, the lengths Tensor inferred by this function
        would be [2, 1, 1, 1, 3] indicating variable batch_size
        dim_1 violates the existing assumption / precondition
        that KeyedJaggedTensor's should have fixed batch_size dimension.

        """
        kjt_keys = list(jt_dict.keys())
        kjt_vals_list: List[torch.Tensor] = []
        kjt_lens_list: List[torch.Tensor] = []
        kjt_weights_list: List[torch.Tensor] = []
        stride_per_key: List[int] = []
        for jt in jt_dict.values():
            stride_per_key.append(len(jt.lengths()))
            kjt_vals_list.append(jt.values())
            kjt_lens_list.append(jt.lengths())
            weight = jt.weights_or_none()
            if weight is not None:
                kjt_weights_list.append(weight)
        kjt_vals = torch.concat(kjt_vals_list)
        kjt_lens = torch.concat(kjt_lens_list)
        kjt_weights = (
            torch.concat(kjt_weights_list) if len(kjt_weights_list) > 0 else None
        )
        kjt_stride, kjt_stride_per_key_per_rank = (
            (stride_per_key[0], None)
            if all(s == stride_per_key[0] for s in stride_per_key)
            else (None, [[stride] for stride in stride_per_key])
        )
        kjt = KeyedJaggedTensor(
            keys=kjt_keys,
            values=kjt_vals,
            weights=kjt_weights,
            lengths=kjt_lens,
            stride=kjt_stride,
            stride_per_key_per_rank=kjt_stride_per_key_per_rank,
        ).sync()
        return kjt

    def sync(self) -> "KeyedJaggedTensor":
        if not is_torchdynamo_compiling():
            self.length_per_key()
            self.offset_per_key()
        return self

    def unsync(self) -> "KeyedJaggedTensor":
        self._length_per_key = None
        self._offset_per_key = None
        return self

    def device(self) -> torch.device:
        return self._values.device

    def lengths(self) -> torch.Tensor:
        _lengths = _maybe_compute_lengths(self._lengths, self._offsets)
        self._lengths = _lengths
        return _lengths

    def lengths_or_none(self) -> Optional[torch.Tensor]:
        return self._lengths

    def offsets(self) -> torch.Tensor:
        _offsets = _maybe_compute_offsets(self._lengths, self._offsets)
        self._offsets = _offsets
        return _offsets

    def offsets_or_none(self) -> Optional[torch.Tensor]:
        return self._offsets

    def keys(self) -> List[str]:
        return self._keys

    def values(self) -> torch.Tensor:
        return self._values

    def weights(self) -> torch.Tensor:
        return _get_weights_or_throw(self._weights)

    def weights_or_none(self) -> Optional[torch.Tensor]:
        return self._weights

    def stride(self) -> int:
        return self._stride

    def stride_per_key(self) -> List[int]:
        return self._stride_per_key

    def stride_per_key_per_rank(self) -> List[List[int]]:
        return self._stride_per_key_per_rank

    def variable_stride_per_key(self) -> bool:
        return self._variable_stride_per_key

    def inverse_indices(self) -> Tuple[List[str], torch.Tensor]:
        return _get_inverse_indices_or_throw(self._inverse_indices)

    def inverse_indices_or_none(self) -> Optional[Tuple[List[str], torch.Tensor]]:
        return self._inverse_indices

    def _key_indices(self) -> Dict[str, int]:
        _index_per_key: Dict[str, int] = _maybe_compute_index_per_key(
            self._keys,
            self._index_per_key,
        )
        self._index_per_key = _index_per_key
        return _index_per_key

    def length_per_key(self) -> List[int]:
        _length_per_key = _maybe_compute_length_per_key(
            keys=self._keys,
            stride=self.stride(),
            stride_per_key=self.stride_per_key(),
            variable_stride_per_key=self.variable_stride_per_key(),
            length_per_key=self._length_per_key,
            lengths=self._lengths,
            offsets=self._offsets,
            values=self._values,
        )
        self._length_per_key = _length_per_key
        return _length_per_key

    def length_per_key_or_none(self) -> Optional[List[int]]:
        return self._length_per_key

    def offset_per_key(self) -> List[int]:
        _length_per_key, _offset_per_key = _maybe_compute_offset_per_key(
            keys=self._keys,
            stride=self.stride(),
            stride_per_key=self.stride_per_key(),
            variable_stride_per_key=self.variable_stride_per_key(),
            length_per_key=self._length_per_key,
            offset_per_key=self._offset_per_key,
            lengths=self._lengths,
            offsets=self._offsets,
            values=self._values,
        )
        self._length_per_key = _length_per_key
        self._offset_per_key = _offset_per_key
        return _offset_per_key

    def offset_per_key_or_none(self) -> Optional[List[int]]:
        return self._offset_per_key

    def lengths_offset_per_key(self) -> List[int]:
        if not self._lengths_offset_per_key:
            self._lengths_offset_per_key = _cumsum(self.stride_per_key())
        return self._lengths_offset_per_key

    def index_per_key(self) -> Dict[str, int]:
        return self._key_indices()

    def split(self, segments: List[int]) -> List["KeyedJaggedTensor"]:
        split_list: List[KeyedJaggedTensor] = []
        start = 0
        start_offset = 0
        _length_per_key = self.length_per_key()
        _offset_per_key = self.offset_per_key()
        for segment in segments:
            end = start + segment
            end_offset = _offset_per_key[end]
            keys: List[str] = self._keys[start:end]
            stride, stride_per_key_per_rank = (
                (None, self.stride_per_key_per_rank()[start:end])
                if self.variable_stride_per_key()
                else (self._stride, None)
            )
            if segment == len(self._keys):
                # no torch slicing required
                split_list.append(
                    KeyedJaggedTensor(
                        keys=self._keys,
                        values=self._values,
                        weights=self.weights_or_none(),
                        lengths=self._lengths,
                        offsets=self._offsets,
                        stride=stride,
                        stride_per_key_per_rank=stride_per_key_per_rank,
                        length_per_key=self._length_per_key,
                        offset_per_key=self._offset_per_key,
                        index_per_key=self._index_per_key,
                        jt_dict=self._jt_dict,
                        inverse_indices=None,
                    )
                )
            elif segment == 0:
                empty_int_list: List[int] = torch.jit.annotate(List[int], [])
                split_list.append(
                    KeyedJaggedTensor(
                        keys=keys,
                        values=torch.tensor(
                            empty_int_list,
                            device=self.device(),
                            dtype=self._values.dtype,
                        ),
                        weights=(
                            None
                            if self.weights_or_none() is None
                            else torch.tensor(
                                empty_int_list,
                                device=self.device(),
                                dtype=self.weights().dtype,
                            )
                        ),
                        lengths=torch.tensor(
                            empty_int_list, device=self.device(), dtype=torch.int
                        ),
                        offsets=torch.tensor(
                            empty_int_list, device=self.device(), dtype=torch.int
                        ),
                        stride=stride,
                        stride_per_key_per_rank=stride_per_key_per_rank,
                        length_per_key=None,
                        offset_per_key=None,
                        index_per_key=None,
                        jt_dict=None,
                        inverse_indices=None,
                    )
                )
            else:
                split_length_per_key = _length_per_key[start:end]

                if not torch.jit.is_scripting() and is_non_strict_exporting():
                    sz = sum(split_length_per_key)

                    [torch._check_is_size(length) for length in split_length_per_key]
                    torch._check(start_offset <= self._values.size(0))
                    torch._check(sz <= self._values.size(0))
                    torch._check_is_size(start_offset)

                    torch._check(start_offset + sz <= self._values.size(0))

                    lengths_start = self.lengths_offset_per_key()[start]
                    lengths_sz = self.lengths_offset_per_key()[end] - lengths_start

                    _lengths = torch.narrow(
                        self.lengths(), 0, lengths_start, lengths_sz
                    )

                    if self.weights_or_none() is not None:
                        torch._check(start_offset + sz <= self.weights().size(0))
                        torch._check(start_offset <= self.weights().size(0))

                    split_list.append(
                        KeyedJaggedTensor(
                            keys=keys,
                            values=torch.narrow(self._values, 0, start_offset, sz),
                            weights=(
                                None
                                if self.weights_or_none() is None
                                else torch.narrow(self.weights(), 0, start_offset, sz)
                            ),
                            lengths=_lengths,
                            offsets=None,
                            stride=stride,
                            stride_per_key_per_rank=stride_per_key_per_rank,
                            length_per_key=split_length_per_key,
                            offset_per_key=None,
                            index_per_key=None,
                            jt_dict=None,
                            inverse_indices=None,
                        )
                    )
                else:
                    pt2_checks_tensor_slice(self._values, start_offset, end_offset)

                    lengths_offset_per_key: List[int] = self.lengths_offset_per_key()
                    pt2_checks_tensor_slice(
                        self.lengths(),
                        lengths_offset_per_key[start],
                        lengths_offset_per_key[end],
                    )

                    split_list.append(
                        KeyedJaggedTensor(
                            keys=keys,
                            values=self._values[start_offset:end_offset],
                            weights=(
                                None
                                if self.weights_or_none() is None
                                else self.weights()[start_offset:end_offset]
                            ),
                            lengths=self.lengths()[
                                lengths_offset_per_key[start] : lengths_offset_per_key[
                                    end
                                ]
                            ],
                            offsets=None,
                            stride=stride,
                            stride_per_key_per_rank=stride_per_key_per_rank,
                            length_per_key=split_length_per_key,
                            offset_per_key=None,
                            index_per_key=None,
                            jt_dict=None,
                            inverse_indices=None,
                        )
                    )
            start = end
            start_offset = end_offset
        return split_list

    def permute(
        self, indices: List[int], indices_tensor: Optional[torch.Tensor] = None
    ) -> "KeyedJaggedTensor":

        if indices_tensor is None:
            indices_tensor = torch.tensor(
                indices, dtype=torch.int, device=self.device()
            )

        length_per_key = self.length_per_key()
        permuted_keys: List[str] = []
        permuted_stride_per_key_per_rank: List[List[int]] = []
        permuted_length_per_key: List[int] = []
        permuted_length_per_key_sum = 0
        for index in indices:
            key = self.keys()[index]
            permuted_keys.append(key)
            permuted_stride_per_key_per_rank.append(
                self.stride_per_key_per_rank()[index]
            )
            permuted_length_per_key.append(length_per_key[index])

        permuted_length_per_key_sum = sum(permuted_length_per_key)
        if not torch.jit.is_scripting() and is_non_strict_exporting():
            torch._check_is_size(permuted_length_per_key_sum)
            torch._check(permuted_length_per_key_sum != -1)
            torch._check(permuted_length_per_key_sum != 0)

        if self.variable_stride_per_key():
            length_per_key_tensor = _pin_and_move(
                torch.tensor(self.length_per_key()), self.device()
            )
            stride_per_key_tensor = _pin_and_move(
                torch.tensor(self.stride_per_key()), self.device()
            )
            permuted_lengths, _ = _permute_tensor_by_segments(
                self.lengths(),
                stride_per_key_tensor,
                indices_tensor,
                None,
            )
            permuted_values, permuted_weights = _permute_tensor_by_segments(
                self.values(),
                length_per_key_tensor,
                indices_tensor,
                self.weights_or_none(),
            )
        else:

            (
                permuted_lengths,
                permuted_values,
                permuted_weights,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                indices_tensor,
                self.lengths().view(len(self._keys), -1),
                self.values(),
                self.weights_or_none(),
                permuted_length_per_key_sum,
            )
        stride, optional_permuted_stride_per_key_per_rank = (
            (None, permuted_stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )
        kjt = KeyedJaggedTensor(
            keys=permuted_keys,
            values=permuted_values,
            weights=permuted_weights,
            lengths=permuted_lengths.view(-1),
            offsets=None,
            stride=stride,
            stride_per_key_per_rank=optional_permuted_stride_per_key_per_rank,
            length_per_key=permuted_length_per_key if len(permuted_keys) > 0 else None,
            offset_per_key=None,
            index_per_key=None,
            jt_dict=None,
            inverse_indices=None,
        )
        return kjt

    def flatten_lengths(self) -> "KeyedJaggedTensor":
        stride, stride_per_key_per_rank = (
            (None, self.stride_per_key_per_rank())
            if self.variable_stride_per_key()
            else (self._stride, None)
        )
        return KeyedJaggedTensor(
            keys=self._keys,
            values=self._values,
            weights=self._weights,
            lengths=self.lengths().view(-1),
            offsets=None,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            length_per_key=self.length_per_key(),
            offset_per_key=None,
            index_per_key=None,
            jt_dict=None,
            inverse_indices=None,
        )

    def __getitem__(self, key: str) -> JaggedTensor:
        offset_per_key = self.offset_per_key()
        index = self._key_indices()[key]
        start_offset = offset_per_key[index]
        end_offset = (
            offset_per_key[index + 1]
            if index + 1 < len(offset_per_key)
            else start_offset
        )

        if not torch.jit.is_scripting() and is_non_strict_exporting():
            length_per_key = self.length_per_key()
            _lengths = torch.narrow(
                self.lengths(),
                0,
                self.lengths_offset_per_key()[index],
                self.lengths_offset_per_key()[index + 1]
                - self.lengths_offset_per_key()[index],
            )
            sz = length_per_key[index]

            torch._check_is_size(start_offset)
            torch._check_is_size(sz)
            torch._check(start_offset <= self.values().size(0))
            torch._check(sz <= self.values().size(0))

            if self.weights_or_none() is not None:
                torch._check(start_offset <= self.weights().size(0))
                torch._check(sz <= self.weights().size(0))

            return JaggedTensor(
                values=torch.narrow(
                    self.values(),
                    0,
                    start_offset,
                    sz,
                ),
                weights=(
                    None
                    if self.weights_or_none() is None
                    else torch.narrow(
                        self.weights(),
                        0,
                        start_offset,
                        sz,
                    )
                ),
                lengths=_lengths,
                offsets=None,
            )
        else:
            pt2_checks_tensor_slice(self._values, start_offset, end_offset)

            return JaggedTensor(
                values=self._values[start_offset:end_offset],
                weights=(
                    None
                    if self.weights_or_none() is None
                    else self.weights()[start_offset:end_offset]
                ),
                lengths=self.lengths()[
                    self.lengths_offset_per_key()[
                        index
                    ] : self.lengths_offset_per_key()[index + 1]
                ],
                offsets=None,
            )

    def to_dict(self) -> Dict[str, JaggedTensor]:
        _jt_dict = _maybe_compute_kjt_to_jt_dict(
            stride=self.stride(),
            stride_per_key=self.stride_per_key(),
            keys=self.keys(),
            length_per_key=self.length_per_key(),
            lengths=self.lengths(),
            values=self.values(),
            variable_stride_per_key=self.variable_stride_per_key(),
            weights=self.weights_or_none(),
            jt_dict=self._jt_dict,
        )
        self._jt_dict = _jt_dict
        return _jt_dict

    @torch.jit.unused
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        self._values.record_stream(stream)
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        if weights is not None:
            weights.record_stream(stream)
        if lengths is not None:
            lengths.record_stream(stream)
        if offsets is not None:
            offsets.record_stream(stream)

    def to(
        self,
        device: torch.device,
        non_blocking: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "KeyedJaggedTensor":
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride, stride_per_key_per_rank = (
            (None, self._stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )
        length_per_key = self._length_per_key
        offset_per_key = self._offset_per_key
        index_per_key = self._index_per_key
        jt_dict = self._jt_dict
        inverse_indices = self._inverse_indices
        if inverse_indices is not None:
            inverse_indices = (
                inverse_indices[0],
                inverse_indices[1].to(device, non_blocking=non_blocking),
            )
        if weights is not None:
            if dtype is not None:
                weights = weights.to(
                    dtype=dtype, device=device, non_blocking=non_blocking
                )
            else:
                weights = weights.to(device=device, non_blocking=non_blocking)

        return KeyedJaggedTensor(
            keys=self._keys,
            values=self._values.to(device, non_blocking=non_blocking),
            weights=weights,
            lengths=(
                lengths.to(device, non_blocking=non_blocking)
                if lengths is not None
                else None
            ),
            offsets=(
                offsets.to(device, non_blocking=non_blocking)
                if offsets is not None
                else None
            ),
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            length_per_key=length_per_key,
            offset_per_key=offset_per_key,
            index_per_key=index_per_key,
            jt_dict=jt_dict,
            inverse_indices=inverse_indices,
        )

    def __str__(self) -> str:
        if len(self._keys) == 0 or self._offsets is None and self._lengths is None:
            return "KeyedJaggedTensor()\n"
        offsets = self.offsets()

        return (
            "KeyedJaggedTensor({\n"
            + ",\n".join(
                [
                    "    "
                    + _jagged_tensor_string(
                        self._keys[index],
                        self._values,
                        self._weights,
                        offsets,
                        sum(self.stride_per_key()[:index]),
                        sum(self.stride_per_key()[: index + 1]),
                    )
                    for index in range(len(self._keys))
                ]
            )
            + "\n})\n"
        )

    def pin_memory(self) -> "KeyedJaggedTensor":
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride, stride_per_key_per_rank = (
            (None, self._stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )
        inverse_indices = self._inverse_indices
        if inverse_indices is not None:
            inverse_indices = (inverse_indices[0], inverse_indices[1].pin_memory())

        return KeyedJaggedTensor(
            keys=self._keys,
            values=self._values.pin_memory(),
            weights=weights.pin_memory() if weights is not None else None,
            lengths=lengths.pin_memory() if lengths is not None else None,
            offsets=offsets.pin_memory() if offsets is not None else None,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            length_per_key=self._length_per_key,
            offset_per_key=self._offset_per_key,
            index_per_key=self._index_per_key,
            jt_dict=None,
            inverse_indices=inverse_indices,
        )

    def dist_labels(self) -> List[str]:
        labels = ["lengths", "values"]
        if self.variable_stride_per_key():
            labels.append("strides")
        if self.weights_or_none() is not None:
            labels.append("weights")
        return labels

    def dist_splits(self, key_splits: List[int]) -> List[List[int]]:
        batch_size_per_split = _sum_by_splits(self.stride_per_key(), key_splits)
        length_per_split = _sum_by_splits(self.length_per_key(), key_splits)
        splits = [batch_size_per_split, length_per_split]
        if self.variable_stride_per_key():
            splits.append(key_splits)
        if self.weights_or_none() is not None:
            splits.append(length_per_split)
        return splits

    def dist_tensors(self) -> List[torch.Tensor]:
        tensors = [self.lengths(), self.values()]
        if self.variable_stride_per_key():
            strides = _pin_and_move(torch.tensor(self.stride_per_key()), self.device())
            tensors.append(strides)
        if self.weights_or_none() is not None:
            tensors.append(self.weights())
        return tensors

    @staticmethod
    def dist_init(
        keys: List[str],
        tensors: List[torch.Tensor],
        variable_stride_per_key: bool,
        num_workers: int,
        recat: Optional[torch.Tensor],
        stride_per_rank: Optional[List[int]],
        stagger: int = 1,
    ) -> "KeyedJaggedTensor":
        assert len(tensors) in [2, 3, 4]
        lengths = tensors[0]
        values = tensors[1]
        stride_per_rank_per_key = tensors[2] if variable_stride_per_key else None
        weights = (
            tensors[-1]
            if (variable_stride_per_key and len(tensors) == 4)
            or (not variable_stride_per_key and len(tensors) == 3)
            else None
        )

        if variable_stride_per_key:
            assert stride_per_rank_per_key is not None
            stride_per_key_per_rank_tensor: torch.Tensor = stride_per_rank_per_key.view(
                num_workers, len(keys)
            ).T.cpu()

            strides_cumsum: torch.Tensor = (
                torch.ops.fbgemm.asynchronous_complete_cumsum(stride_per_rank_per_key)
            ).cpu()

            cumsum_lengths = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

            n = strides_cumsum.size(0)
            strides_cumsum_from_1 = torch.narrow(
                strides_cumsum, dim=0, start=1, length=n - 1
            )
            strides_cumsum_to_minus_1 = torch.narrow(
                strides_cumsum, dim=0, start=0, length=n - 1
            )
            length_per_key_tensor = (
                cumsum_lengths[strides_cumsum_from_1]
                - cumsum_lengths[strides_cumsum_to_minus_1]
            )

            with record_function("## all2all_data:recat_values ##"):
                if recat is not None:
                    lengths, _ = _permute_tensor_by_segments(
                        lengths,
                        stride_per_rank_per_key,
                        torch.jit._unwrap_optional(recat),
                        None,
                    )
                    values, weights = _permute_tensor_by_segments(
                        values,
                        length_per_key_tensor,
                        torch.jit._unwrap_optional(recat),
                        weights,
                    )

            stride_per_key_per_rank = torch.jit.annotate(
                List[List[int]], stride_per_key_per_rank_tensor.tolist()
            )

            if not stride_per_key_per_rank:
                stride_per_key_per_rank = [[0]] * len(keys)
            if stagger > 1:
                stride_per_key_per_rank_stagger: List[List[int]] = []
                local_world_size = num_workers // stagger
                for i in range(len(keys)):
                    stride_per_rank_stagger: List[int] = []
                    for j in range(local_world_size):
                        stride_per_rank_stagger.extend(
                            stride_per_key_per_rank[i][j::local_world_size]
                        )
                    stride_per_key_per_rank_stagger.append(stride_per_rank_stagger)
                stride_per_key_per_rank = stride_per_key_per_rank_stagger

            kjt = KeyedJaggedTensor(
                keys=keys,
                values=values,
                weights=weights,
                lengths=lengths,
                stride_per_key_per_rank=stride_per_key_per_rank,
            )
            return kjt.sync()
        else:
            assert stride_per_rank is not None
            with record_function("## all2all_data:recat_values ##"):
                if recat is not None:
                    stride = stride_per_rank[0]

                    single_batch_per_rank = True
                    if not is_torchdynamo_compiling():
                        single_batch_per_rank = all(
                            s == stride for s in stride_per_rank
                        )

                    if single_batch_per_rank:
                        (
                            lengths,
                            values,
                            weights,
                        ) = torch.ops.fbgemm.permute_2D_sparse_data(
                            torch.jit._unwrap_optional(recat),
                            lengths.view(-1, stride),
                            values,
                            weights,
                            values.numel(),
                        )
                        lengths = lengths.view(-1)
                    else:  # variable batch size per rank
                        (
                            lengths,
                            values,
                            weights,
                        ) = torch.ops.fbgemm.permute_1D_sparse_data(
                            torch.jit._unwrap_optional(recat),
                            lengths.view(-1),
                            values,
                            weights,
                            values.numel(),
                        )
            kjt = KeyedJaggedTensor(
                keys=keys,
                values=values,
                weights=weights,
                lengths=lengths,
                stride=sum(stride_per_rank),
            )
            return kjt.sync()

class KeyedTensor(Pipelineable, metaclass=JaggedTensorMeta):
    """
    KeyedTensor holds a concatenated list of dense tensors, each of which can be
    accessed by a key.

    The keyed dimension can be of variable length (length_per_key).
    Common use cases uses include storage of pooled embeddings of different dimensions.

    Implementation is torch.jit.script-able.

    Args:
        keys (List[str]): list of keys.
        length_per_key (List[int]): length of each key along key dimension.
        values (torch.Tensor): dense tensor, concatenated typically along key dimension.
        key_dim (int): key dimension, zero indexed - defaults to 1
            (typically B is 0-dimension).

    Example::

        # kt is KeyedTensor holding

        #                         0           1           2
        #     "Embedding A"    [1,1]       [1,1]        [1,1]
        #     "Embedding B"    [2,1,2]     [2,1,2]      [2,1,2]
        #     "Embedding C"    [3,1,2,3]   [3,1,2,3]    [3,1,2,3]

        tensor_list = [
            torch.tensor([[1,1]] * 3),
            torch.tensor([[2,1,2]] * 3),
            torch.tensor([[3,1,2,3]] * 3),
        ]

        keys = ["Embedding A", "Embedding B", "Embedding C"]

        kt = KeyedTensor.from_tensor_list(keys, tensor_list)

        kt.values()
            # tensor(
            #     [
            #         [1, 1, 2, 1, 2, 3, 1, 2, 3],
            #         [1, 1, 2, 1, 2, 3, 1, 2, 3],
            #         [1, 1, 2, 1, 2, 3, 1, 2, 3],
            #     ]
            # )

        kt["Embedding B"]
            # tensor([[2, 1, 2], [2, 1, 2], [2, 1, 2]])
    """

    def __init__(
        self,
        keys: List[str],
        length_per_key: List[int],
        values: torch.Tensor,
        key_dim: int = 1,
        # Below exposed to ensure torch.script-able
        offset_per_key: Optional[List[int]] = None,
        index_per_key: Optional[Dict[str, int]] = None,
    ) -> None:
        self._keys = keys
        self._length_per_key = length_per_key
        self._values = values
        self._key_dim = key_dim

        self._offset_per_key: Optional[List[int]] = offset_per_key
        self._index_per_key: Optional[Dict[str, int]] = index_per_key

    @staticmethod
    def from_tensor_list(
        keys: List[str], tensors: List[torch.Tensor], key_dim: int = 1, cat_dim: int = 1
    ) -> "KeyedTensor":
        length_per_key = [tensor.shape[key_dim] for tensor in tensors]
        return KeyedTensor(
            keys=keys,
            length_per_key=length_per_key,
            values=torch.cat(tensors, dim=cat_dim),
            key_dim=key_dim,
        )

    def keys(self) -> List[str]:
        return self._keys

    def values(self) -> torch.Tensor:
        return self._values

    def key_dim(self) -> int:
        return self._key_dim

    def device(self) -> torch.device:
        return self._values.device

    def offset_per_key(self) -> List[int]:
        _offset_per_key = _maybe_compute_offset_per_key_kt(
            self._length_per_key,
            self._offset_per_key,
        )
        self._offset_per_key = _offset_per_key
        return _offset_per_key

    def length_per_key(self) -> List[int]:
        return self._length_per_key

    def _key_indices(self) -> Dict[str, int]:
        _index_per_key = _maybe_compute_index_per_key(
            self._keys,
            self._index_per_key,
        )
        self._index_per_key = _index_per_key
        return _index_per_key

    def __getitem__(self, key: str) -> torch.Tensor:
        index = self._key_indices()[key]
        start = self.offset_per_key()[index]
        length = self._length_per_key[index]
        return self._values.narrow(dim=self._key_dim, start=start, length=length)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        indices = self._key_indices()
        lengths = self._length_per_key
        split_values = self._values.split(lengths, dim=self._key_dim)
        return {key: split_values[index] for (key, index) in indices.items()}

    @staticmethod
    def regroup(
        keyed_tensors: List["KeyedTensor"], groups: List[List[str]]
    ) -> List[torch.Tensor]:
        # Fast path, one-to-one correspondence between keyed_tensors and groups
        if _all_keys_used_once(keyed_tensors, groups) is True:
            return _fbgemm_permute_pooled_embs(keyed_tensors, groups)
        else:  # Fallback to slow path otherwise
            return _regroup_keyed_tensors(keyed_tensors, groups)

    @staticmethod
    def regroup_as_dict(
        keyed_tensors: List["KeyedTensor"], groups: List[List[str]], keys: List[str]
    ) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = {}
        assert len(groups) == len(keys), "Groups and keys should have same length"
        tensor_list = KeyedTensor.regroup(keyed_tensors, groups)
        for i, key in enumerate(keys):
            ret[key] = tensor_list[i]
        return ret

    @torch.jit.unused
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        self._values.record_stream(stream)

    def to(self, device: torch.device, non_blocking: bool = False) -> "KeyedTensor":
        return KeyedTensor(
            keys=self._keys,
            length_per_key=self._length_per_key,
            values=self._values.to(device, non_blocking=non_blocking),
            key_dim=self._key_dim,
            offset_per_key=self._offset_per_key,
            index_per_key=self._index_per_key,
        )

    def __str__(self) -> str:
        if len(self._keys) == 0:
            return "KeyedTensor()\n"

        return (
            "KeyedTensor({\n"
            + ",\n".join(
                [
                    '    "{}": '.format(key) + _keyed_values_string(self[key])
                    for key in self._keys
                ]
            )
            + "\n})\n"
        )

class JaggedTensor(Pipelineable, metaclass=JaggedTensorMeta):
    """
    Represents an (optionally weighted) jagged tensor.

    A `JaggedTensor` is a tensor with a *jagged dimension* which is dimension whose
    slices may be of different lengths. See `KeyedJaggedTensor` for full example.

    Implementation is torch.jit.script-able.

    NOTE:
        We will NOT do input validation as it's expensive, you should always pass in the
        valid lengths, offsets, etc.

    Args:
        values (torch.Tensor): values tensor in dense representation.
        weights (Optional[torch.Tensor]): if values have weights. Tensor with same shape
            as values.
        lengths (Optional[torch.Tensor]): jagged slices, represented as lengths.
        offsets (Optional[torch.Tensor]): jagged slices, represented as cumulative
            offsets.
    """

    _fields = ["_values", "_weights", "_lengths", "_offsets"]

    def __init__(
        self,
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> None:

        self._values: torch.Tensor = values
        self._weights: Optional[torch.Tensor] = weights
        _assert_offsets_or_lengths_is_provided(offsets, lengths)
        if offsets is not None:
            _assert_tensor_has_no_elements_or_has_integers(offsets, "offsets")
        if lengths is not None:
            _assert_tensor_has_no_elements_or_has_integers(lengths, "lengths")
        self._lengths: Optional[torch.Tensor] = lengths
        self._offsets: Optional[torch.Tensor] = offsets

    @staticmethod
    def empty(
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        values_dtype: Optional[torch.dtype] = None,
        weights_dtype: Optional[torch.dtype] = None,
        lengths_dtype: torch.dtype = torch.int32,
    ) -> "JaggedTensor":
        weights = (
            torch.empty(0, dtype=weights_dtype, device=device) if is_weighted else None
        )
        return JaggedTensor(
            values=torch.empty(0, dtype=values_dtype, device=device),
            offsets=torch.empty(0, dtype=lengths_dtype, device=device),
            lengths=torch.empty(0, dtype=lengths_dtype, device=device),
            weights=weights,
        )

    @staticmethod
    def from_dense_lengths(
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> "JaggedTensor":
        """
        Constructs `JaggedTensor` from dense values/weights of shape (B, N,).

        Note that `lengths` is still of shape (B,).
        """

        mask2d = (
            _arange(end=values.size(1), device=values.device).expand(values.size(0), -1)
        ) < lengths.unsqueeze(-1)
        return JaggedTensor(
            values=values[mask2d],
            weights=_optional_mask(weights, mask2d),
            lengths=lengths,
        )

    @staticmethod
    def from_dense(
        values: List[torch.Tensor],
        weights: Optional[List[torch.Tensor]] = None,
    ) -> "JaggedTensor":
        """
        Constructs `JaggedTensor` from dense values/weights of shape (B, N,).

        Note that `lengths` and `offsets` are still of shape (B,).

        Args:
            values (List[torch.Tensor]): a list of tensors for dense representation
            weights (Optional[List[torch.Tensor]]): if values have weights, tensor with
                the same shape as values.

        Returns:
            JaggedTensor: JaggedTensor created from 2D dense tensor.

        Example::

            values = [
                torch.Tensor([1.0]),
                torch.Tensor(),
                torch.Tensor([7.0, 8.0]),
                torch.Tensor([10.0, 11.0, 12.0]),
            ]
            weights = [
                torch.Tensor([1.0]),
                torch.Tensor(),
                torch.Tensor([7.0, 8.0]),
                torch.Tensor([10.0, 11.0, 12.0]),
            ]
            j1 = JaggedTensor.from_dense(
                values=values,
                weights=weights,
            )

            # j1 = [[1.0], [], [7.0], [8.0], [10.0, 11.0, 12.0]]
        """

        values_tensor = torch.cat(values, dim=0)
        lengths = torch.tensor(
            [value.size(0) for value in values],
            dtype=torch.int32,
            device=values_tensor.device,
        )
        weights_tensor = torch.cat(weights, dim=0) if weights is not None else None

        return JaggedTensor(
            values=values_tensor,
            weights=weights_tensor,
            lengths=lengths,
        )

    def to_dense(self) -> List[torch.Tensor]:
        """
        Constructs a dense-representation of the JT's values.

        Returns:
            List[torch.Tensor]: list of tensors.

        Example::

            values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
            jt = JaggedTensor(values=values, offsets=offsets)

            values_list = jt.to_dense()

            # values_list = [
            #     torch.tensor([1.0, 2.0]),
            #     torch.tensor([]),
            #     torch.tensor([3.0]),
            #     torch.tensor([4.0]),
            #     torch.tensor([5.0]),
            #     torch.tensor([6.0, 7.0, 8.0]),
            # ]
        """
        tensor_list = []
        for index in range(self.offsets().size(0) - 1):
            offset = self.offsets()[index].item()
            next_offset = self.offsets()[index + 1].item()
            tensor_list.append(self.values()[offset:next_offset])
        return tensor_list

    def to_dense_weights(self) -> Optional[List[torch.Tensor]]:
        """
        Constructs a dense-representation of the JT's weights.

        Returns:
            Optional[List[torch.Tensor]]: list of tensors, `None` if no weights.

        Example::

            values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
            jt = JaggedTensor(values=values, weights=weights, offsets=offsets)

            weights_list = jt.to_dense_weights()

            # weights_list = [
            #     torch.tensor([0.1, 0.2]),
            #     torch.tensor([]),
            #     torch.tensor([0.3]),
            #     torch.tensor([0.4]),
            #     torch.tensor([0.5]),
            #     torch.tensor([0.6, 0.7, 0.8]),
            # ]
        """
        if self.weights_or_none() is None:
            return None
        tensor_list = []
        for index in range(self.offsets().size(0) - 1):
            offset = self.offsets()[index].item()
            next_offset = self.offsets()[index + 1].item()
            tensor_list.append(self.weights()[offset:next_offset])
        return tensor_list

    def to_padded_dense(
        self,
        desired_length: Optional[int] = None,
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Constructs a 2D dense tensor from the JT's values of shape (B, N,).

        Note that `B` is the length of self.lengths() and `N` is the longest feature
        length or `desired_length`.

        If `desired_length` > `length` we will pad with `padding_value`, otherwise we
        will select the last value at `desired_length`.

        Args:
            desired_length (int): the length of the tensor.
            padding_value (float): padding value if we need to pad.

        Returns:
            torch.Tensor: 2d dense tensor.

        Example::

            values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
            jt = JaggedTensor(values=values, offsets=offsets)

            dt = jt.to_padded_dense(
                desired_length=2,
                padding_value=10.0,
            )

            # dt = [
            #     [1.0, 2.0],
            #     [10.0, 10.0],
            #     [3.0, 10.0],
            #     [4.0, 10.0],
            #     [5.0, 10.0],
            #     [6.0, 7.0],
            # ]
        """
        if desired_length is None:
            N = int(torch.max(self.lengths()).item())
        else:
            N = desired_length
        return torch.ops.fbgemm.jagged_to_padded_dense(
            self.values(), [self.offsets()], [N], padding_value
        )

    def to_padded_dense_weights(
        self,
        desired_length: Optional[int] = None,
        padding_value: float = 0.0,
    ) -> Optional[torch.Tensor]:
        """
        Constructs a 2D dense tensor from the JT's weights of shape (B, N,).

        Note that `B` is the length of self.lengths() and `N` is the longest feature
        length or `desired_length`.

        If `desired_length` > `length` we will pad with `padding_value`, otherwise we
        will select the last value at `desired_length`.

        Args:
            desired_length (int): the length of the tensor.
            padding_value (float): padding value if we need to pad.

        Returns:
            Optional[torch.Tensor]: 2d dense tensor, `None` if no weights.

        Example::

            values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
            jt = JaggedTensor(values=values, weights=weights, offsets=offsets)

            d_wt = jt.to_padded_dense_weights(
                desired_length=2,
                padding_value=1.0,
            )

            # d_wt = [
            #     [0.1, 0.2],
            #     [1.0, 1.0],
            #     [0.3, 1.0],
            #     [0.4, 1.0],
            #     [0.5, 1.0],
            #     [0.6, 0.7],
            # ]
        """
        if self.weights_or_none() is None:
            return None
        if desired_length is None:
            N = int(torch.max(self.lengths()).item())
        else:
            N = desired_length
        return torch.ops.fbgemm.jagged_to_padded_dense(
            self.weights(), [self.offsets()], [N], padding_value
        )

    def device(self) -> torch.device:
        return self._values.device

    def lengths(self) -> torch.Tensor:
        _lengths = _maybe_compute_lengths(self._lengths, self._offsets)
        self._lengths = _lengths
        return _lengths

    def lengths_or_none(self) -> Optional[torch.Tensor]:
        return self._lengths

    def offsets(self) -> torch.Tensor:
        _offsets = _maybe_compute_offsets(self._lengths, self._offsets)
        self._offsets = _offsets
        return _offsets

    def offsets_or_none(self) -> Optional[torch.Tensor]:
        return self._offsets

    def values(self) -> torch.Tensor:
        return self._values

    def weights(self) -> torch.Tensor:
        return _get_weights_or_throw(self._weights)

    def weights_or_none(self) -> Optional[torch.Tensor]:
        return self._weights

    def to(self, device: torch.device, non_blocking: bool = False) -> "JaggedTensor":
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        return JaggedTensor(
            values=self._values.to(device, non_blocking=non_blocking),
            weights=(
                weights.to(device, non_blocking=non_blocking)
                if weights is not None
                else None
            ),
            lengths=(
                lengths.to(device, non_blocking=non_blocking)
                if lengths is not None
                else None
            ),
            offsets=(
                offsets.to(device, non_blocking=non_blocking)
                if offsets is not None
                else None
            ),
        )

    @torch.jit.unused
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        self._values.record_stream(stream)
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        if weights is not None:
            weights.record_stream(stream)
        if lengths is not None:
            lengths.record_stream(stream)
        if offsets is not None:
            offsets.record_stream(stream)

    def __str__(self) -> str:
        offsets = self.offsets()

        if self._weights is None:
            return (
                "JaggedTensor({\n    "
                + _jagged_values_string(self._values, offsets, 0, len(offsets) - 1)
                + "\n})\n"
            )

        return (
            "JaggedTensor({\n"
            + '    "values": '
            + _jagged_values_string(self._values, offsets, 0, len(offsets) - 1)
            + ',\n    "weights": '
            + _jagged_values_string(
                _get_weights_or_throw(self._weights), offsets, 0, len(offsets) - 1
            )
            + "\n})\n"
        )

