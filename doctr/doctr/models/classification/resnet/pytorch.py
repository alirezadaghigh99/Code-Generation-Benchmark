def resnet18(pretrained: bool = False, **kwargs: Any) -> TVResNet:
    """ResNet-18 architecture as described in `"Deep Residual Learning for Image Recognition",
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    >>> import torch
    >>> from doctr.models import resnet18
    >>> model = resnet18(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the ResNet architecture

    Returns:
    -------
        A resnet18 model
    """
    return _tv_resnet(
        "resnet18",
        pretrained,
        tv_resnet18,
        ignore_keys=["fc.weight", "fc.bias"],
        **kwargs,
    )