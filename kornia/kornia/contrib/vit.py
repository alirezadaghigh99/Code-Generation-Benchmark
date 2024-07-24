def from_config(variant: str, pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
        """Build ViT model based on the given config string. The format is ``vit_{size}/{patch_size}``.
        E.g. ``vit_b/16`` means ViT-Base, patch size 16x16. If ``pretrained=True``, AugReg weights are loaded.
        The weights are hosted on HuggingFace's model hub: https://huggingface.co/kornia.

        .. note::
            The available weights are: ``vit_l/16``, ``vit_b/16``, ``vit_s/16``, ``vit_ti/16``,
            ``vit_b/32``, ``vit_s/32``.

        Args:
            variant: ViT model variant e.g. ``vit_b/16``.
            pretrained: whether to load pre-trained AugReg weights.
            kwargs: other keyword arguments that will be passed to :func:`kornia.contrib.vit.VisionTransformer`.
        Returns:
            The respective ViT model

        Example:
            >>> from kornia.contrib import VisionTransformer
            >>> vit_model = VisionTransformer.from_config("vit_b/16", pretrained=True)
        """
        model_type, patch_size_str = variant.split("/")
        patch_size = int(patch_size_str)

        model_config = {
            "vit_ti": {"embed_dim": 192, "depth": 12, "num_heads": 3},
            "vit_s": {"embed_dim": 384, "depth": 12, "num_heads": 6},
            "vit_b": {"embed_dim": 768, "depth": 12, "num_heads": 12},
            "vit_l": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
            "vit_h": {"embed_dim": 1280, "depth": 32, "num_heads": 16},
        }[model_type]
        kwargs.update(model_config, patch_size=patch_size)

        model = VisionTransformer(**kwargs)

        if pretrained:
            url = _get_weight_url(variant)
            state_dict = torch.hub.load_state_dict_from_url(url)
            model.load_state_dict(state_dict)

        return model

class VisionTransformer(Module):
    """Vision transformer (ViT) module.

    The module is expected to be used as operator for different vision tasks.

    The method is inspired from existing implementations of the paper :cite:`dosovitskiy2020vit`.

    .. warning::
        This is an experimental API subject to changes in favor of flexibility.

    Args:
        image_size: the size of the input image.
        patch_size: the size of the patch to compute the embedding.
        in_channels: the number of channels for the input.
        embed_dim: the embedding dimension inside the transformer encoder.
        depth: the depth of the transformer.
        num_heads: the number of attention heads.
        dropout_rate: dropout rate.
        dropout_attn: attention dropout rate.
        backbone: an nn.Module to compute the image patches embeddings.

    Example:
        >>> img = torch.rand(1, 3, 224, 224)
        >>> vit = VisionTransformer(image_size=224, patch_size=16)
        >>> vit(img).shape
        torch.Size([1, 197, 768])
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
        backbone: Module | None = None,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_size = embed_dim

        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, patch_size, image_size, backbone)
        hidden_dim = self.patch_embedding.out_channels
        self.encoder = TransformerEncoder(hidden_dim, depth, num_heads, dropout_rate, dropout_attn)
        self.norm = nn.LayerNorm(hidden_dim, 1e-6)

    @property
    def encoder_results(self) -> list[Tensor]:
        return self.encoder.results

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"Input x type is not a Tensor. Got: {type(x)}")

        if self.image_size not in (*x.shape[-2:],) and x.shape[-3] != self.in_channels:
            raise ValueError(
                f"Input image shape must be Bx{self.in_channels}x{self.image_size}x{self.image_size}. Got: {x.shape}"
            )

        out = self.patch_embedding(x)
        out = self.encoder(out)
        out = self.norm(out)
        return out

    @staticmethod
    def from_config(variant: str, pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
        """Build ViT model based on the given config string. The format is ``vit_{size}/{patch_size}``.
        E.g. ``vit_b/16`` means ViT-Base, patch size 16x16. If ``pretrained=True``, AugReg weights are loaded.
        The weights are hosted on HuggingFace's model hub: https://huggingface.co/kornia.

        .. note::
            The available weights are: ``vit_l/16``, ``vit_b/16``, ``vit_s/16``, ``vit_ti/16``,
            ``vit_b/32``, ``vit_s/32``.

        Args:
            variant: ViT model variant e.g. ``vit_b/16``.
            pretrained: whether to load pre-trained AugReg weights.
            kwargs: other keyword arguments that will be passed to :func:`kornia.contrib.vit.VisionTransformer`.
        Returns:
            The respective ViT model

        Example:
            >>> from kornia.contrib import VisionTransformer
            >>> vit_model = VisionTransformer.from_config("vit_b/16", pretrained=True)
        """
        model_type, patch_size_str = variant.split("/")
        patch_size = int(patch_size_str)

        model_config = {
            "vit_ti": {"embed_dim": 192, "depth": 12, "num_heads": 3},
            "vit_s": {"embed_dim": 384, "depth": 12, "num_heads": 6},
            "vit_b": {"embed_dim": 768, "depth": 12, "num_heads": 12},
            "vit_l": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
            "vit_h": {"embed_dim": 1280, "depth": 32, "num_heads": 16},
        }[model_type]
        kwargs.update(model_config, patch_size=patch_size)

        model = VisionTransformer(**kwargs)

        if pretrained:
            url = _get_weight_url(variant)
            state_dict = torch.hub.load_state_dict_from_url(url)
            model.load_state_dict(state_dict)

        return model

