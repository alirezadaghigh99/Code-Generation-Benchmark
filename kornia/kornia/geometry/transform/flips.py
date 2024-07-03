class Rot180(Module):
    r"""Rotate a tensor image or a batch of tensor images 180 degrees.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input tensor.

    Examples:
        >>> rot180 = Rot180()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> rot180(input)
        tensor([[[[1., 1., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])
    """

    def forward(self, input: Tensor) -> Tensor:
        return rot180(input)

    def __repr__(self) -> str:
        return self.__class__.__name__class Vflip(Module):
    r"""Vertically flip a tensor image or a batch of tensor images.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input tensor.

    Returns:
        The vertically flipped image tensor.

    Examples:
        >>> vflip = Vflip()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> vflip(input)
        tensor([[[[0., 1., 1.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])
    """

    def forward(self, input: Tensor) -> Tensor:
        return vflip(input)

    def __repr__(self) -> str:
        return self.__class__.__name__