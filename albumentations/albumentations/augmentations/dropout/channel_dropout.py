class ChannelDropout(ImageOnlyTransform):
    """Randomly Drop Channels in the input Image.

    Args:
        channel_drop_range (int, int): range from which we choose the number of channels to drop.
        fill_value (int, float): pixel value for the dropped channel.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, unit32, float32

    """

    class InitSchema(BaseTransformInitSchema):
        channel_drop_range: OnePlusIntRangeType = (1, 1)
        fill_value: Annotated[ColorType, Field(description="Pixel value for the dropped channel.")]

    def __init__(
        self,
        channel_drop_range: tuple[int, int] = (1, 1),
        fill_value: float = 0,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        self.channel_drop_range = channel_drop_range
        self.fill_value = fill_value

    def apply(self, img: np.ndarray, channels_to_drop: tuple[int, ...], **params: Any) -> np.ndarray:
        return channel_dropout(img, channels_to_drop, self.fill_value)

    def get_params_dependent_on_targets(self, params: Mapping[str, Any]) -> dict[str, Any]:
        img = params["image"]
        num_channels = img.shape[-1]

        if is_grayscale_image(img):
            msg = "Images has one channel. ChannelDropout is not defined."
            raise NotImplementedError(msg)

        if self.channel_drop_range[1] >= num_channels:
            msg = "Can not drop all channels in ChannelDropout."
            raise ValueError(msg)

        num_drop_channels = random.randint(self.channel_drop_range[0], self.channel_drop_range[1])

        channels_to_drop = random.sample(range(num_channels), k=num_drop_channels)

        return {"channels_to_drop": channels_to_drop}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "channel_drop_range", "fill_value"

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

