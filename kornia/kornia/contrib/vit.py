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

