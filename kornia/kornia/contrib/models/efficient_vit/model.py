class EfficientViTConfig:
    """Configuration to construct EfficientViT model.

    Model weights can be loaded from a checkpoint URL or local path.
    The model weights are hosted on HuggingFace's model hub: https://huggingface.co/kornia.

    Args:
        checkpoint: URL or local path of model weights.
    """

    checkpoint: str = field(default_factory=_get_base_url)

    @classmethod
    def from_pretrained(
        cls, model_type: Literal["b1", "b2", "b3"], resolution: Literal[224, 256, 288]
    ) -> EfficientViTConfig:
        """Return a configuration object from a pre-trained model.

        Args:
            model_type: model type, one of :obj:`"b1"`, :obj:`"b2"`, :obj:`"b3"`.
            resolution: input resolution, one of :obj:`224`, :obj:`256`, :obj:`288`.
        """
        return cls(checkpoint=_get_base_url(model_type=model_type, resolution=resolution))

