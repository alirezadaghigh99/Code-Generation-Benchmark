def resnet50(
    weights: ResNet50_Weights | None = None, *args: Any, **kwargs: Any
) -> ResNet:
    """ResNet-50 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/1512.03385.pdf

    .. versionchanged:: 0.4
       Switched to multi-weight support API.

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :func:`timm.create_model`.
        **kwargs: Additional keywork arguments to pass to :func:`timm.create_model`.

    Returns:
        A ResNet-50 model.
    """
    if weights:
        kwargs['in_chans'] = weights.meta['in_chans']

    model: ResNet = timm.create_model('resnet50', *args, **kwargs)

    if weights:
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        assert set(missing_keys) <= {'fc.weight', 'fc.bias'}
        assert not unexpected_keys

    return model