def extract_tensor_patches(
    input: Tensor,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: PadType = 0,
    allow_auto_padding: bool = False,
) -> Tensor:
    r"""Function that extract patches from tensors and stacks them.

    See :class:`~kornia.contrib.ExtractTensorPatches` for details.

    Args:
        input: tensor image where to extract the patches with shape :math:`(B, C, H, W)`.
        window_size: the size of the sliding window and the output patch size.
        stride: stride of the sliding window.
        padding: Zero-padding added to both side of the input.
        allow_auto_adding: whether to allow automatic padding if the window and stride do not fit into the image.

    Returns:
        the tensor with the extracted patches with shape :math:`(B, N, C, H_{out}, W_{out})`.

    Examples:
        >>> input = torch.arange(9.).view(1, 1, 3, 3)
        >>> patches = extract_tensor_patches(input, (2, 3))
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> patches[:, -1]
        tensor([[[[3., 4., 5.],
                  [6., 7., 8.]]]])
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input input type is not a Tensor. Got {type(input)}")

    if len(input.shape) != 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    # check if the window sliding over the image will fit into the image
    # torch's unfold drops the final patches that don't fit
    window_size = cast(Tuple[int, int], _pair(window_size))
    stride = cast(Tuple[int, int], _pair(stride))
    original_size = (input.shape[-2], input.shape[-1])

    if not padding:
        # if padding is specified, we leave it up to the user to ensure it fits
        # otherwise we check here if it will fit and offer to calculate padding
        if not _check_patch_fit(original_size, window_size, stride):
            if not allow_auto_padding:
                warn(
                    f"The window will not fit into the image. \nWindow size: {window_size}\nStride: {stride}\n"
                    f"Image size: {original_size}\n"
                    "This means that the final incomplete patches will be dropped. By enabling `allow_auto_padding`, "
                    "the input will be padded to fit the window and stride."
                )
            else:
                padding = compute_padding(original_size=original_size, window_size=window_size, stride=stride)

    if padding:
        padding = create_padding_tuple(padding)
        input = pad(input, padding)

    return _extract_tensor_patchesnd(input, window_size, stride)

class ExtractTensorPatches(Module):
    r"""Module that extract patches from tensors and stack them.

    In the simplest case, the output value of the operator with input size
    :math:`(B, C, H, W)` is :math:`(B, N, C, H_{out}, W_{out})`.

    where
      - :math:`B` is the batch size.
      - :math:`N` denotes the total number of extracted patches stacked in
      - :math:`C` denotes the number of input channels.
      - :math:`H`, :math:`W` the input height and width of the input in pixels.
      - :math:`H_{out}`, :math:`W_{out}` denote to denote to the patch size
        defined in the function signature.
        left-right and top-bottom order.

    * :attr:`window_size` is the size of the sliding window and controls the
      shape of the output tensor and defines the shape of the output patch.
    * :attr:`stride` controls the stride to apply to the sliding window and
      regulates the overlapping between the extracted patches.
    * :attr:`padding` controls the amount of implicit zeros-paddings on both
      sizes at each dimension.
    * :attr:`allow_auto_padding` allows automatic calculation of the padding required
      to fit the window and stride into the image.

    The parameters :attr:`window_size`, :attr:`stride` and :attr:`padding` can
    be either:

        - a single ``int`` -- in which case the same value is used for the
          height and width dimension.
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
          the height dimension, and the second `int` for the width dimension.

    :attr:`padding` can also be a ``tuple`` of four ints -- in which case, the
    first two ints are for the height dimension while the last two ints are for
    the width dimension.

    Args:
        input: tensor image where to extract the patches with shape :math:`(B, C, H, W)`.
        window_size: the size of the sliding window and the output patch size.
        stride: stride of the sliding window.
        padding: Zero-padding added to both side of the input.
        allow_auto_adding: whether to allow automatic padding if the window and stride do not fit into the image.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, N, C, H_{out}, W_{out})`

    Returns:
        the tensor with the extracted patches.

    Examples:
        >>> input = torch.arange(9.).view(1, 1, 3, 3)
        >>> patches = extract_tensor_patches(input, (2, 3))
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> patches[:, -1]
        tensor([[[[3., 4., 5.],
                  [6., 7., 8.]]]])
    """

    def __init__(
        self,
        window_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: PadType = 0,
        allow_auto_padding: bool = False,
    ) -> None:
        super().__init__()
        self.window_size: Union[int, Tuple[int, int]] = window_size
        self.stride: Union[int, Tuple[int, int]] = stride
        self.padding: PadType = padding
        self.allow_auto_padding: bool = allow_auto_padding

    def forward(self, input: Tensor) -> Tensor:
        return extract_tensor_patches(
            input,
            self.window_size,
            stride=self.stride,
            padding=self.padding,
            allow_auto_padding=self.allow_auto_padding,
        )

class CombineTensorPatches(Module):
    r"""Module that combines patches back into full tensors.

    In the simplest case, the output value of the operator with input size
    :math:`(B, N, C, H_{out}, W_{out})` is :math:`(B, C, H, W)`.

    where
      - :math:`B` is the batch size.
      - :math:`N` denotes the total number of extracted patches stacked in
      - :math:`C` denotes the number of input channels.
      - :math:`H`, :math:`W` the input height and width of the input in pixels.
      - :math:`H_{out}`, :math:`W_{out}` denote to denote to the patch size
        defined in the function signature.
        left-right and top-bottom order.


    * :attr:`original_size` is the size of the original image prior to
      extracting tensor patches and defines the shape of the output patch.
    * :attr:`window_size` is the size of the sliding window used while
      extracting tensor patches.
    * :attr:`stride` controls the stride to apply to the sliding window and
      regulates the overlapping between the extracted patches.
    * :attr:`unpadding` is the amount of padding to be removed. If specified,
      this value must be the same as padding used while extracting tensor patches.
    * :attr:`allow_auto_unpadding` allows automatic calculation of the padding required
      to fit the window and stride into the image. This must be used if the
      `allow_auto_padding` flag was used for extracting the patches.


    The parameters :attr:`original_size`, :attr:`window_size`, :attr:`stride`, and :attr:`unpadding` can
    be either:

        - a single ``int`` -- in which case the same value is used for the
          height and width dimension.
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
          the height dimension, and the second `int` for the width dimension.

    :attr:`unpadding` can also be a ``tuple`` of four ints -- in which case, the
    first two ints are for the height dimension while the last two ints are for
    the width dimension.

    Args:
        patches: patched tensor with shape :math:`(B, N, C, H_{out}, W_{out})`.
        original_size: the size of the original tensor and the output size.
        window_size: the size of the sliding window used while extracting patches.
        stride: stride of the sliding window.
        unpadding: remove the padding added to both side of the input.
        allow_auto_unpadding: whether to allow automatic unpadding of the input
            if the window and stride do not fit into the original_size.
        eps: small value used to prevent division by zero.

    Shape:
        - Input: :math:`(B, N, C, H_{out}, W_{out})`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> out = extract_tensor_patches(torch.arange(16).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2))
        >>> combine_tensor_patches(out, original_size=(4, 4), window_size=(2, 2), stride=(2, 2))
        tensor([[[[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11],
                  [12, 13, 14, 15]]]])

    .. note::
        This function is supposed to be used in conjunction with :class:`ExtractTensorPatches`.
    """

    def __init__(
        self,
        original_size: Tuple[int, int],
        window_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        unpadding: PadType = 0,
        allow_auto_unpadding: bool = False,
    ) -> None:
        super().__init__()
        self.original_size: Tuple[int, int] = original_size
        self.window_size: Union[int, Tuple[int, int]] = window_size
        self.stride: Union[int, Tuple[int, int]] = stride if stride is not None else window_size
        self.unpadding: PadType = unpadding
        self.allow_auto_unpadding: bool = allow_auto_unpadding

    def forward(self, input: Tensor) -> Tensor:
        return combine_tensor_patches(
            input,
            self.original_size,
            self.window_size,
            stride=self.stride,
            unpadding=self.unpadding,
            allow_auto_unpadding=self.allow_auto_unpadding,
        )

