class MaxPool1d(_MaxPoolNd):
    r"""Applies a 1D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`
    and output :math:`(N, C, L_{out})` can be precisely described as:

    .. math::
        out(N_i, C_j, k) = \max_{m=0, \ldots, \text{kernel\_size} - 1}
                input(N_i, C_j, stride \times k + m)

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` is the stride between the elements within the
    sliding window. This `link`_ has a nice visualization of the pooling parameters.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    Args:
        kernel_size: The size of the sliding window, must be > 0.
        stride: The stride of the sliding window, must be > 0. Default value is :attr:`kernel_size`.
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        return_indices: If ``True``, will return the argmax along with the max values.
                        Useful for :class:`torch.nn.MaxUnpool1d` later
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Examples::

        >>> # pool of size=3, stride=2
        >>> m = nn.MaxPool1d(3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    dilation: _size_1_t

    def forward(self, input: Tensor):
        return F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )

class MaxPool2d(_MaxPoolNd):
    r"""Applies a 2D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def forward(self, input: Tensor):
        return F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )

class MaxPool3d(_MaxPoolNd):
    r"""Applies a 3D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \max_{k=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                              & \text{input}(N_i, C_j, \text{stride[0]} \times d + k,
                                                             \text{stride[1]} \times h + m, \text{stride[2]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on all three sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool3d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times
                (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times
                (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times
                (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50, 44, 31)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """  # noqa: E501

    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    dilation: _size_3_t

    def forward(self, input: Tensor):
        return F.max_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )

class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    r"""Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1})` or :math:`(C, S_{0}, S_{1})`, where
          :math:`S=\text{output\_size}`.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveAvgPool2d((5, 7))
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveAvgPool2d((None, 7))
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)

    """

    output_size: _size_2_opt_t

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(input, self.output_size)

class AdaptiveAvgPool3d(_AdaptiveAvgPoolNd):
    r"""Applies a 3D adaptive average pooling over an input signal composed of several input planes.

    The output is of size D x H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the form D x H x W.
                     Can be a tuple (D, H, W) or a single number D for a cube D x D x D.
                     D, H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1}, S_{2})` or :math:`(C, S_{0}, S_{1}, S_{2})`,
          where :math:`S=\text{output\_size}`.

    Examples:
        >>> # target output size of 5x7x9
        >>> m = nn.AdaptiveAvgPool3d((5, 7, 9))
        >>> input = torch.randn(1, 64, 8, 9, 10)
        >>> output = m(input)
        >>> # target output size of 7x7x7 (cube)
        >>> m = nn.AdaptiveAvgPool3d(7)
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)
        >>> # target output size of 7x9x8
        >>> m = nn.AdaptiveAvgPool3d((7, None, None))
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)

    """

    output_size: _size_3_opt_t

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool3d(input, self.output_size)

class AdaptiveMaxPool1d(_AdaptiveMaxPoolNd):
    r"""Applies a 1D adaptive max pooling over an input signal composed of several input planes.

    The output size is :math:`L_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size :math:`L_{out}`.
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool1d. Default: ``False``

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where
          :math:`L_{out}=\text{output\_size}`.

    Examples:
        >>> # target output size of 5
        >>> m = nn.AdaptiveMaxPool1d(5)
        >>> input = torch.randn(1, 64, 8)
        >>> output = m(input)

    """

    output_size: _size_1_t

    def forward(self, input: Tensor):
        return F.adaptive_max_pool1d(input, self.output_size, self.return_indices)

class AvgPool1d(_AvgPoolNd):
    r"""Applies a 1D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`,
    output :math:`(N, C, L_{out})` and :attr:`kernel_size` :math:`k`
    can be precisely described as:

    .. math::

        \text{out}(N_i, C_j, l) = \frac{1}{k} \sum_{m=0}^{k-1}
                               \text{input}(N_i, C_j, \text{stride} \times l + m)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can each be
    an ``int`` or a one-element tuple.

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} +
              2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} + 1\right\rfloor

          Per the note above, if ``ceil_mode`` is True and :math:`(L_{out} - 1) \times \text{stride} \geq L_{in}
          + \text{padding}`, we skip the last window as it would start in the right padded region, resulting in
          :math:`L_{out}` being reduced by one.

    Examples::

        >>> # pool with window of size=3, stride=2
        >>> m = nn.AvgPool1d(3, stride=2)
        >>> m(torch.tensor([[[1., 2, 3, 4, 5, 6, 7]]]))
        tensor([[[2., 4., 6.]]])
    """

    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t = None,
        padding: _size_1_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )

class AvgPool2d(_AvgPoolNd):
    r"""Applies a 2D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used.


    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          Per the note above, if ``ceil_mode`` is True and :math:`(H_{out} - 1)\times \text{stride}[0]\geq H_{in}
          + \text{padding}[0]`, we skip the last window as it would start in the bottom padded region,
          resulting in :math:`H_{out}` being reduced by one.

          The same applies for :math:`W_{out}`.

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool2d((3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)
    """

    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )

class FractionalMaxPool2d(Module):
    r"""Applies a 2D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    .. note:: Exactly one of ``output_size`` or ``output_ratio`` must be defined.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number k (for a square kernel of k x k) or a tuple `(kh, kw)`
        output_size: the target output size of the image of the form `oH x oW`.
                     Can be a tuple `(oH, oW)` or a single number oH for a square image `oH x oH`.
                     Note that we must have :math:`kH + oH - 1 <= H_{in}` and :math:`kW + oW - 1 <= W_{in}`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1).
                      Note that we must have :math:`kH + (output\_ratio\_H * H_{in}) - 1 <= H_{in}`
                      and :math:`kW + (output\_ratio\_W * W_{in}) - 1 <= W_{in}`
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :meth:`nn.MaxUnpool2d`. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
          :math:`(H_{out}, W_{out})=\text{output\_size}` or
          :math:`(H_{out}, W_{out})=\text{output\_ratio} \times (H_{in}, W_{in})`.

    Examples:
        >>> # pool of square window of size=3, and target output size 13x12
        >>> m = nn.FractionalMaxPool2d(3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> m = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _Fractional MaxPooling:
        https://arxiv.org/abs/1412.6071
    """

    __constants__ = ["kernel_size", "return_indices", "output_size", "output_ratio"]

    kernel_size: _size_2_t
    return_indices: bool
    output_size: _size_2_t
    output_ratio: _ratio_2_t

    def __init__(
        self,
        kernel_size: _size_2_t,
        output_size: Optional[_size_2_t] = None,
        output_ratio: Optional[_ratio_2_t] = None,
        return_indices: bool = False,
        _random_samples=None,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.return_indices = return_indices
        self.register_buffer("_random_samples", _random_samples)
        self.output_size = _pair(output_size) if output_size is not None else None
        self.output_ratio = _pair(output_ratio) if output_ratio is not None else None
        if output_size is None and output_ratio is None:
            raise ValueError(
                "FractionalMaxPool2d requires specifying either "
                "an output size, or a pooling ratio"
            )
        if output_size is not None and output_ratio is not None:
            raise ValueError(
                "only one of output_size and output_ratio may be specified"
            )
        if self.output_ratio is not None:
            if not (0 < self.output_ratio[0] < 1 and 0 < self.output_ratio[1] < 1):
                raise ValueError(
                    f"output_ratio must be between 0 and 1 (got {output_ratio})"
                )

    def forward(self, input: Tensor):
        return F.fractional_max_pool2d(
            input,
            self.kernel_size,
            self.output_size,
            self.output_ratio,
            self.return_indices,
            _random_samples=self._random_samples,
        )

class AvgPool3d(_AvgPoolNd):
    r"""Applies a 3D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \sum_{k=0}^{kD-1} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1} \\
                                              & \frac{\text{input}(N_i, C_j, \text{stride}[0] \times d + k,
                                                      \text{stride}[1] \times h + m, \text{stride}[2] \times w + n)}
                                                     {kD \times kH \times kW}
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on all three sides
    for :attr:`padding` number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on all three sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise :attr:`kernel_size` will be used

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or
          :math:`(C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] -
                    \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] -
                    \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] -
                    \text{kernel\_size}[2]}{\text{stride}[2]} + 1\right\rfloor

          Per the note above, if ``ceil_mode`` is True and :math:`(D_{out} - 1)\times \text{stride}[0]\geq D_{in}
          + \text{padding}[0]`, we skip the last window as it would start in the padded region,
          resulting in :math:`D_{out}` being reduced by one.

          The same applies for :math:`W_{out}` and :math:`H_{out}`.

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50, 44, 31)
        >>> output = m(input)
    """

    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]

    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )

    def __setstate__(self, d):
        super().__setstate__(d)
        self.__dict__.setdefault("padding", 0)
        self.__dict__.setdefault("ceil_mode", False)
        self.__dict__.setdefault("count_include_pad", True)

