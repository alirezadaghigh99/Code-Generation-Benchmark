class Denormalize(nn.Module):
    r"""Denormalize a tensor image with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] * std[channel]) + mean[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Shape:
        - Input: Image tensor of size :math:`(*, C, ...)`.
        - Output: Denormalised tensor with same size as input :math:`(*, C, ...)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = Denormalize(0.0, 255.)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std = 255. * torch.ones(1, 4)
        >>> out = Denormalize(mean, std)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3, 3])
    """

    def __init__(self, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) -> None:
        super().__init__()

        self.mean = mean
        self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return denormalize(input, self.mean, self.std)

    def __repr__(self) -> str:
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr

class Normalize(nn.Module):
    r"""Normalize a tensor image with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Shape:
        - Input: Image tensor of size :math:`(*, C, ...)`.
        - Output: Normalised tensor with same size as input :math:`(*, C, ...)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = Normalize(0.0, 255.)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3)
        >>> mean = torch.zeros(4)
        >>> std = 255. * torch.ones(4)
        >>> out = Normalize(mean, std)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """

    def __init__(
        self,
        mean: Union[torch.Tensor, Tuple[float], List[float], float],
        std: Union[torch.Tensor, Tuple[float], List[float], float],
    ) -> None:
        super().__init__()

        if isinstance(mean, (int, float)):
            mean = torch.tensor([mean])

        if isinstance(std, (int, float)):
            std = torch.tensor([std])

        if isinstance(mean, (tuple, list)):
            mean = torch.tensor(mean)

        if isinstance(std, (tuple, list)):
            std = torch.tensor(std)

        self.mean = mean
        self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return normalize(input, self.mean, self.std)

    def __repr__(self) -> str:
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr

class Normalize(nn.Module):
    r"""Normalize a tensor image with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Shape:
        - Input: Image tensor of size :math:`(*, C, ...)`.
        - Output: Normalised tensor with same size as input :math:`(*, C, ...)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = Normalize(0.0, 255.)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3)
        >>> mean = torch.zeros(4)
        >>> std = 255. * torch.ones(4)
        >>> out = Normalize(mean, std)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """

    def __init__(
        self,
        mean: Union[torch.Tensor, Tuple[float], List[float], float],
        std: Union[torch.Tensor, Tuple[float], List[float], float],
    ) -> None:
        super().__init__()

        if isinstance(mean, (int, float)):
            mean = torch.tensor([mean])

        if isinstance(std, (int, float)):
            std = torch.tensor([std])

        if isinstance(mean, (tuple, list)):
            mean = torch.tensor(mean)

        if isinstance(std, (tuple, list)):
            std = torch.tensor(std)

        self.mean = mean
        self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return normalize(input, self.mean, self.std)

    def __repr__(self) -> str:
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr

class Denormalize(nn.Module):
    r"""Denormalize a tensor image with mean and standard deviation.

    .. math::
        \text{input[channel] = (input[channel] * std[channel]) + mean[channel]}

    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,

    Args:
        mean: Mean for each channel.
        std: Standard deviations for each channel.

    Shape:
        - Input: Image tensor of size :math:`(*, C, ...)`.
        - Output: Denormalised tensor with same size as input :math:`(*, C, ...)`.

    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = Denormalize(0.0, 255.)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x = torch.rand(1, 4, 3, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std = 255. * torch.ones(1, 4)
        >>> out = Denormalize(mean, std)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3, 3])
    """

    def __init__(self, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) -> None:
        super().__init__()

        self.mean = mean
        self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return denormalize(input, self.mean, self.std)

    def __repr__(self) -> str:
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr

