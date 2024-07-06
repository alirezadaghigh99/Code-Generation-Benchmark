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

