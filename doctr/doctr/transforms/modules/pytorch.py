class RandomResize(torch.nn.Module):
    """Randomly resize the input image and align corresponding targets

    >>> import torch
    >>> from doctr.transforms import RandomResize
    >>> transfo = RandomResize((0.3, 0.9), preserve_aspect_ratio=True, symmetric_pad=True, p=0.5)
    >>> out = transfo(torch.rand((3, 64, 64)))

    Args:
    ----
        scale_range: range of the resizing factor for width and height (independently)
        preserve_aspect_ratio: whether to preserve the aspect ratio of the image,
            given a float value, the aspect ratio will be preserved with this probability
        symmetric_pad: whether to symmetrically pad the image,
            given a float value, the symmetric padding will be applied with this probability
        p: probability to apply the transformation
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.3, 0.9),
        preserve_aspect_ratio: Union[bool, float] = False,
        symmetric_pad: Union[bool, float] = False,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.scale_range = scale_range
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.p = p
        self._resize = Resize

    def forward(self, img: torch.Tensor, target: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        if torch.rand(1) < self.p:
            scale_h = np.random.uniform(*self.scale_range)
            scale_w = np.random.uniform(*self.scale_range)
            new_size = (int(img.shape[-2] * scale_h), int(img.shape[-1] * scale_w))

            _img, _target = self._resize(
                new_size,
                preserve_aspect_ratio=self.preserve_aspect_ratio
                if isinstance(self.preserve_aspect_ratio, bool)
                else bool(torch.rand(1) <= self.symmetric_pad),
                symmetric_pad=self.symmetric_pad
                if isinstance(self.symmetric_pad, bool)
                else bool(torch.rand(1) <= self.symmetric_pad),
            )(img, target)

            return _img, _target
        return img, target

    def extra_repr(self) -> str:
        return f"scale_range={self.scale_range}, preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}, p={self.p}"  # noqa: E501