def nested_tensor(tensor_list, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) -> Tensor:
    r"""
Constructs a nested tensor with no autograd history (also known as a "leaf tensor", see
:ref:`Autograd mechanics <autograd-mechanics>`) from :attr:`tensor_list` a list of tensors.

Args:
    tensor_list (List[array_like]): a list of tensors, or anything that can be passed to torch.tensor,
    where each element of the list has the same dimensionality.

Keyword arguments:
    dtype (:class:`torch.dtype`, optional): the desired type of returned nested tensor.
        Default: if None, same :class:`torch.dtype` as leftmost tensor in the list.
    layout (:class:`torch.layout`, optional): the desired layout of returned nested tensor.
        Only strided and jagged layouts are supported. Default: if None, the strided layout.
    device (:class:`torch.device`, optional): the desired device of returned nested tensor.
        Default: if None, same :class:`torch.device` as leftmost tensor in the list
    requires_grad (bool, optional): If autograd should record operations on the
        returned nested tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned nested tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.

Example::

    >>> a = torch.arange(3, dtype=torch.float, requires_grad=True)
    >>> b = torch.arange(5, dtype=torch.float, requires_grad=True)
    >>> nt = torch.nested.nested_tensor([a, b], requires_grad=True)
    >>> nt.is_leaf
    True
    """
    if layout is None:
        layout = torch.strided
    if layout == torch.strided:
        return _nested.nested_tensor(
            tensor_list,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory)
    elif layout == torch.jagged:
        # Need to wrap lists of scalars as tensors
        list_of_tensors = [t if isinstance(t, Tensor) else torch.as_tensor(t) for t in tensor_list]

        from torch.nested._internal.nested_tensor import jagged_from_list

        with torch.no_grad():
            nt, _ = jagged_from_list(list_of_tensors, offsets=None, device=device, dtype=dtype)

        nt.requires_grad_(requires_grad)
        if pin_memory:
            nt = nt.pin_memory()  # type: ignore[assignment]

        return nt
    else:
        raise RuntimeError(f"Specified layout is unsupported for nested tensors: {layout}")

def nested_tensor_from_jagged(
    values: Tensor,
    offsets: Optional[Tensor] = None,
    lengths: Optional[Tensor] = None,
    jagged_dim: Optional[int] = None,
) -> Tensor:
    r"""
Constructs a jagged layout nested tensor from the given jagged components. The jagged layout
consists of a required values buffer with the jagged dimension packed into a single dimension.
The offsets / lengths metadata determines how this dimension is split into batch elements
and are expected to be allocated on the same device as the values buffer.

Expected metadata formats:
    * offsets: Indices within the packed dimension splitting it into heterogeneously-sized
      batch elements. Example: [0, 2, 3, 6] indicates that a packed jagged dim of size 6
      should be conceptually split into batch elements of length [2, 1, 3]. Note that both the
      beginning and ending offsets are required for kernel convenience (i.e. shape batch_size + 1).
    * lengths: Lengths of the individual batch elements; shape == batch_size. Example: [2, 1, 3]
      indicates that a packed jagged dim of size 6 should be conceptually split into batch
      elements of length [2, 1, 3].

Note that it can be useful to provide both offsets and lengths. This describes a nested tensor
with "holes", where the offsets indicate the start position of each batch item and the length
specifies the total number of elements (see example below).

The returned jagged layout nested tensor will be a view of the input values tensor.

Args:
    values (:class:`torch.Tensor`): The underlying buffer in the shape of
        (sum_B(*), D_1, ..., D_N). The jagged dimension is packed into a single dimension,
        with the offsets / lengths metadata used to distinguish batch elements.
    offsets (optional :class:`torch.Tensor`): Offsets into the jagged dimension of shape B + 1.
    lengths (optional :class:`torch.Tensor`): Lengths of the batch elements of shape B.
    jagged_dim (optional int): Indicates which dimension in values is the packed jagged
        dimension. If None, this is set to dim=1 (i.e. the dimension immediately following
        the batch dimension). Default: None

Example::

    >>> values = torch.randn(12, 5)
    >>> offsets = torch.tensor([0, 3, 5, 6, 10, 12])
    >>> nt = nested_tensor_from_jagged(values, offsets)
    >>> # 3D shape with the middle dimension jagged
    >>> nt.shape
    torch.Size([5, j2, 5])
    >>> # Length of each item in the batch:
    >>> offsets.diff()
    tensor([3, 2, 1, 4, 2])

    >>> values = torch.randn(6, 5)
    >>> offsets = torch.tensor([0, 2, 3, 6])
    >>> lengths = torch.tensor([1, 1, 2])
    >>> # NT with holes
    >>> nt = nested_tensor_from_jagged(values, offsets, lengths)
    >>> a, b, c = nt.unbind()
    >>> # Batch item 1 consists of indices [0, 1)
    >>> torch.equal(a, values[0:1, :])
    True
    >>> # Batch item 2 consists of indices [2, 3)
    >>> torch.equal(b, values[2:3, :])
    True
    >>> # Batch item 3 consists of indices [3, 5)
    >>> torch.equal(c, values[3:5, :])
    True
    """
    if offsets is None:
        if lengths is None:
            raise RuntimeError(
                "nested_tensor_from_jagged(): At least one of offsets or lengths is required."
            )
        else:
            # TODO: Truly support offsets=None at some point?
            # For now, just convert lengths -> offsets for kernel convenience
            offsets = F.pad(lengths.cumsum(0), (1, 0))
            lengths = None

    if jagged_dim is None:
        jagged_dim = 1

    from torch.nested._internal.nested_tensor import nested_view_from_values_offsets_lengths

    return nested_view_from_values_offsets_lengths(values, offsets, lengths, ragged_idx=jagged_dim)

