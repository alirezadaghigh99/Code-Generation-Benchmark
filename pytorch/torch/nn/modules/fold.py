class Fold(Module):
    r"""Combines an array of sliding local blocks into a large containing tensor.

    Consider a batched :attr:`input` tensor containing sliding local blocks,
    e.g., patches of images, of shape :math:`(N, C \times  \prod(\text{kernel\_size}), L)`,
    where :math:`N` is batch dimension, :math:`C \times \prod(\text{kernel\_size})`
    is the number of values within a block (a block has :math:`\prod(\text{kernel\_size})`
    spatial locations each containing a :math:`C`-channeled vector), and
    :math:`L` is the total number of blocks. (This is exactly the
    same specification as the output shape of :class:`~torch.nn.Unfold`.) This
    operation combines these local blocks into the large :attr:`output` tensor
    of shape :math:`(N, C, \text{output\_size}[0], \text{output\_size}[1], \dots)`
    by summing the overlapping values. Similar to :class:`~torch.nn.Unfold`, the
    arguments must satisfy

    .. math::
        L = \prod_d \left\lfloor\frac{\text{output\_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    where :math:`d` is over all spatial dimensions.

    * :attr:`output_size` describes the spatial shape of the large containing
      tensor of the sliding local blocks. It is useful to resolve the ambiguity
      when multiple input shapes map to same number of sliding blocks, e.g.,
      with ``stride > 0``.

    The :attr:`padding`, :attr:`stride` and :attr:`dilation` arguments specify
    how the sliding blocks are retrieved.

    * :attr:`stride` controls the stride for the sliding blocks.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension before
      reshaping.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the \u00e0 trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Args:
        output_size (int or tuple): the shape of the spatial dimensions of the
                                    output (i.e., ``output.sizes()[2:]``)
        kernel_size (int or tuple): the size of the sliding blocks
        dilation (int or tuple, optional): a parameter that controls the
                                           stride of elements within the
                                           neighborhood. Default: 1
        padding (int or tuple, optional): implicit zero padding to be added on
                                          both sides of input. Default: 0
        stride (int or tuple): the stride of the sliding blocks in the input
                               spatial dimensions. Default: 1

    * If :attr:`output_size`, :attr:`kernel_size`, :attr:`dilation`,
      :attr:`padding` or :attr:`stride` is an int or a tuple of length 1 then
      their values will be replicated across all spatial dimensions.

    * For the case of two output spatial dimensions this operation is sometimes
      called ``col2im``.

    .. note::
        :class:`~torch.nn.Fold` calculates each combined value in the resulting
        large tensor by summing all values from all containing blocks.
        :class:`~torch.nn.Unfold` extracts the values in the local blocks by
        copying from the large tensor. So, if the blocks overlap, they are not
        inverses of each other.

        In general, folding and unfolding operations are related as
        follows. Consider :class:`~torch.nn.Fold` and
        :class:`~torch.nn.Unfold` instances created with the same
        parameters:

        >>> fold_params = dict(kernel_size=..., dilation=..., padding=..., stride=...)
        >>> fold = nn.Fold(output_size=..., **fold_params)
        >>> unfold = nn.Unfold(**fold_params)

        Then for any (supported) ``input`` tensor the following
        equality holds:

        ::

            fold(unfold(input)) == divisor * input

        where ``divisor`` is a tensor that depends only on the shape
        and dtype of the ``input``:

        >>> # xdoctest: +SKIP
        >>> input_ones = torch.ones(input.shape, dtype=input.dtype)
        >>> divisor = fold(unfold(input_ones))

        When the ``divisor`` tensor contains no zero elements, then
        ``fold`` and ``unfold`` operations are inverses of each
        other (up to constant divisor).

    .. warning::
        Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.

    Shape:
        - Input: :math:`(N, C \times \prod(\text{kernel\_size}), L)` or :math:`(C \times \prod(\text{kernel\_size}), L)`
        - Output: :math:`(N, C, \text{output\_size}[0], \text{output\_size}[1], \dots)`
          or :math:`(C, \text{output\_size}[0], \text{output\_size}[1], \dots)` as described above

    Examples::

        >>> fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 2))
        >>> input = torch.randn(1, 3 * 2 * 2, 12)
        >>> output = fold(input)
        >>> output.size()
        torch.Size([1, 3, 4, 5])

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    """

    __constants__ = ["output_size", "kernel_size", "dilation", "padding", "stride"]
    output_size: _size_any_t
    kernel_size: _size_any_t
    dilation: _size_any_t
    padding: _size_any_t
    stride: _size_any_t

    def __init__(
        self,
        output_size: _size_any_t,
        kernel_size: _size_any_t,
        dilation: _size_any_t = 1,
        padding: _size_any_t = 0,
        stride: _size_any_t = 1,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        return F.fold(
            input,
            self.output_size,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )

    def extra_repr(self) -> str:
        return (
            "output_size={output_size}, kernel_size={kernel_size}, "
            "dilation={dilation}, padding={padding}, stride={stride}".format(
                **self.__dict__
            )
        )

