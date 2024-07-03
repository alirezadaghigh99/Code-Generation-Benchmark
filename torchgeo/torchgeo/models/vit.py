def vit_small_patch16_224(
    weights: ViTSmall16_Weights | None = None, *args: Any, **kwargs: Any
) -> VisionTransformer:
    """Vision Transform (ViT) small patch size 16 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2010.11929

    .. versionadded:: 0.4

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :func:`timm.create_model`.
        **kwargs: Additional keywork arguments to pass to :func:`timm.create_model`.

    Returns:
        A ViT small 16 model.
    """
    if weights:
        kwargs['in_chans'] = weights.meta['in_chans']

    model: VisionTransformer = timm.create_model(
        'vit_small_patch16_224', *args, **kwargs
    )

    if weights:
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        assert set(missing_keys) <= {'head.weight', 'head.bias'}
        assert not unexpected_keys

    return model