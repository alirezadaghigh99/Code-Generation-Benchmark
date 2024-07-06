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

