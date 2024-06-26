class Resize(T.Resize):
    """Resize the input image to the given size"""

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation=F.InterpolationMode.BILINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        super().__init__(size, interpolation, antialias=True)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad

        if not isinstance(self.size, (int, tuple, list)):
            raise AssertionError("size should be either a tuple, a list or an int")

    def forward(
        self,
        img: torch.Tensor,
        target: Optional[np.ndarray] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
        if isinstance(self.size, int):
            target_ratio = img.shape[-2] / img.shape[-1]
        else:
            target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[-2] / img.shape[-1]

        if not self.preserve_aspect_ratio or (target_ratio == actual_ratio and (isinstance(self.size, (tuple, list)))):
            # If we don't preserve the aspect ratio or the wanted aspect ratio is the same than the original one
            # We can use with the regular resize
            if target is not None:
                return super().forward(img), target
            return super().forward(img)
        else:
            # Resize
            if isinstance(self.size, (tuple, list)):
                if actual_ratio > target_ratio:
                    tmp_size = (self.size[0], max(int(self.size[0] / actual_ratio), 1))
                else:
                    tmp_size = (max(int(self.size[1] * actual_ratio), 1), self.size[1])
            elif isinstance(self.size, int):  # self.size is the longest side, infer the other
                if img.shape[-2] <= img.shape[-1]:
                    tmp_size = (max(int(self.size * actual_ratio), 1), self.size)
                else:
                    tmp_size = (self.size, max(int(self.size / actual_ratio), 1))

            # Scale image
            img = F.resize(img, tmp_size, self.interpolation, antialias=True)
            raw_shape = img.shape[-2:]
            if isinstance(self.size, (tuple, list)):
                # Pad (inverted in pytorch)
                _pad = (0, self.size[1] - img.shape[-1], 0, self.size[0] - img.shape[-2])
                if self.symmetric_pad:
                    half_pad = (math.ceil(_pad[1] / 2), math.ceil(_pad[3] / 2))
                    _pad = (half_pad[0], _pad[1] - half_pad[0], half_pad[1], _pad[3] - half_pad[1])
                img = pad(img, _pad)

            # In case boxes are provided, resize boxes if needed (for detection task if preserve aspect ratio)
            if target is not None:
                if self.preserve_aspect_ratio:
                    # Get absolute coords
                    if target.shape[1:] == (4,):
                        if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                            if np.max(target) <= 1:
                                offset = half_pad[0] / img.shape[-1], half_pad[1] / img.shape[-2]
                            target[:, [0, 2]] = offset[0] + target[:, [0, 2]] * raw_shape[-1] / img.shape[-1]
                            target[:, [1, 3]] = offset[1] + target[:, [1, 3]] * raw_shape[-2] / img.shape[-2]
                        else:
                            target[:, [0, 2]] *= raw_shape[-1] / img.shape[-1]
                            target[:, [1, 3]] *= raw_shape[-2] / img.shape[-2]
                    elif target.shape[1:] == (4, 2):
                        if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                            if np.max(target) <= 1:
                                offset = half_pad[0] / img.shape[-1], half_pad[1] / img.shape[-2]
                            target[..., 0] = offset[0] + target[..., 0] * raw_shape[-1] / img.shape[-1]
                            target[..., 1] = offset[1] + target[..., 1] * raw_shape[-2] / img.shape[-2]
                        else:
                            target[..., 0] *= raw_shape[-1] / img.shape[-1]
                            target[..., 1] *= raw_shape[-2] / img.shape[-2]
                    else:
                        raise AssertionError
                return img, target

            return img

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        _repr = f"output_size={self.size}, interpolation='{interpolate_str}'"
        if self.preserve_aspect_ratio:
            _repr += f", preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}"
        return f"{self.__class__.__name__}({_repr})"