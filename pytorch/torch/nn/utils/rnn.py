def pad_sequence(
    sequences: Union[Tensor, List[Tensor]],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> Tensor:
    r"""Pad a list of variable length Tensors with :attr:`padding_value`.

    ``pad_sequence`` stacks a list of Tensors along a new dimension, and pads them
    to equal length. :attr:`sequences` can be list of sequences with size ``L x *``,
    where `L` is length of the sequence and ``*`` is any number of dimensions
    (including 0). If :attr:`batch_first` is ``False``, the output is of size
    ``T x B x *``, and ``B x T x *`` otherwise, where ``B`` is the batch size
    (the number of elements in :attr:`sequences`), ``T`` is the length of the longest
    sequence.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise.
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """
    if not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        # JIT doesn't support `Iterable`
        if not isinstance(sequences, Iterable):
            msg = (
                "pad_sequence: Expected iterable for input sequences, but got arg of type: "
                f"{type(sequences)}"
            )
            raise RuntimeError(msg)

        # In JIT context this leads to,
        # RuntimeError: cannot statically infer the expected size of a list in this context
        sequences = tuple(sequences)
    else:
        # For JIT, we only support Union[Tensor, Tuple[Tensor]]
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.unbind(0)

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    return torch._C._nn.pad_sequence(sequences, batch_first, padding_value)

class PackedSequence(PackedSequence_):
    r"""Holds the data and list of :attr:`batch_sizes` of a packed sequence.

    All RNN modules accept packed sequences as inputs.

    Note:
        Instances of this class should never be created manually. They are meant
        to be instantiated by functions like :func:`pack_padded_sequence`.

        Batch sizes represent the number elements at each sequence step in
        the batch, not the varying sequence lengths passed to
        :func:`pack_padded_sequence`.  For instance, given data ``abc`` and ``x``
        the :class:`PackedSequence` would contain data ``axbc`` with
        ``batch_sizes=[2,1,1]``.

    Attributes:
        data (Tensor): Tensor containing packed sequence
        batch_sizes (Tensor): Tensor of integers holding
            information about the batch size at each sequence step
        sorted_indices (Tensor, optional): Tensor of integers holding how this
            :class:`PackedSequence` is constructed from sequences.
        unsorted_indices (Tensor, optional): Tensor of integers holding how this
            to recover the original sequences with correct order.

    .. note::
        :attr:`data` can be on arbitrary device and of arbitrary dtype.
        :attr:`sorted_indices` and :attr:`unsorted_indices` must be ``torch.int64``
        tensors on the same device as :attr:`data`.

        However, :attr:`batch_sizes` should always be a CPU ``torch.int64`` tensor.

        This invariant is maintained throughout :class:`PackedSequence` class,
        and all functions that construct a :class:`PackedSequence` in PyTorch
        (i.e., they only pass in tensors conforming to this constraint).

    """

    def __new__(
        cls, data, batch_sizes=None, sorted_indices=None, unsorted_indices=None
    ):
        return super().__new__(
            cls,
            *_packed_sequence_init_args(
                data, batch_sizes, sorted_indices, unsorted_indices
            ),
        )

    # NOTE [ device and dtype of a PackedSequence ]
    #
    # See the note above in doc string (starting with ":attr:`data` can be on
    # arbitrary device...").
    def pin_memory(self):
        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(
            self.data.pin_memory(),
            self.batch_sizes,
            bind(self.sorted_indices, lambda t: t.pin_memory()),
            bind(self.unsorted_indices, lambda t: t.pin_memory()),
        )

    def cuda(self, *args, **kwargs):
        # Tests to see if 'cuda' should be added to kwargs
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(
            *args, **kwargs
        )
        if ex.is_cuda:
            return self.to(*args, **kwargs)
        return self.to(*args, device="cuda", **kwargs)

    def cpu(self, *args, **kwargs):
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(
            *args, **kwargs
        )
        if ex.device.type == "cpu":
            return self.to(*args, **kwargs)
        return self.to(*args, device="cpu", **kwargs)

    def double(self):
        return self.to(dtype=torch.double)

    def float(self):
        return self.to(dtype=torch.float)

    def half(self):
        return self.to(dtype=torch.half)

    def long(self):
        return self.to(dtype=torch.long)

    def int(self):
        return self.to(dtype=torch.int)

    def short(self):
        return self.to(dtype=torch.short)

    def char(self):
        return self.to(dtype=torch.int8)

    def byte(self):
        return self.to(dtype=torch.uint8)

    def to(self, *args, **kwargs):
        r"""Perform dtype and/or device conversion on `self.data`.

        It has similar signature as :meth:`torch.Tensor.to`, except optional
        arguments like `non_blocking` and `copy` should be passed as kwargs,
        not args, or they will not apply to the index tensors.

        .. note::

            If the ``self.data`` Tensor already has the correct :class:`torch.dtype`
            and :class:`torch.device`, then ``self`` is returned.
            Otherwise, returns a copy with the desired configuration.
        """
        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        data = self.data.to(*args, **kwargs)
        if data is self.data:
            return self
        else:
            # Does not forward device or dtype arg/kwargs, device is set from data.device
            kwargs = dict(
                filter(lambda t: t[0] != "device" and t[0] != "dtype", kwargs.items())
            )
            sorted_indices = bind(
                self.sorted_indices, lambda t: t.to(data.device, **kwargs)
            )
            unsorted_indices = bind(
                self.unsorted_indices, lambda t: t.to(data.device, **kwargs)
            )
            return type(self)(data, self.batch_sizes, sorted_indices, unsorted_indices)

    @property
    def is_cuda(self):
        r"""Return true if `self.data` stored on a gpu."""
        return self.data.is_cuda

    def is_pinned(self):
        r"""Return true if `self.data` stored on in pinned memory."""
        return self.data.is_pinned()

