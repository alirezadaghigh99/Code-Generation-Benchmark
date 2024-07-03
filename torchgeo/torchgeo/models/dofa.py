def dofa_large_patch16_224(
    weights: DOFALarge16_Weights | None = None, *args: Any, **kwargs: Any
) -> DOFA:
    """Dynamic One-For-All (DOFA) large patch size 16 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2403.15356

    .. versionadded:: 0.6

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :class:`DOFA`.
        **kwargs: Additional keywork arguments to pass to :class:`DOFA`.

    Returns:
        A DOFA large 16 model.
    """
    kwargs |= {'patch_size': 16, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16}
    model = DOFA(*args, **kwargs)

    if weights:
        missing_keys, unexpected_keys = model.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        # Both fc_norm and head are generated dynamically
        assert set(missing_keys) <= {
            'fc_norm.weight',
            'fc_norm.bias',
            'head.weight',
            'head.bias',
        }
        assert not unexpected_keys

    return model