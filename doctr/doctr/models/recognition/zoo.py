def recognition_predictor(arch: Any = "crnn_vgg16_bn", pretrained: bool = False, **kwargs: Any) -> RecognitionPredictor:
    """Text recognition architecture.

    Example::
        >>> import numpy as np
        >>> from doctr.models import recognition_predictor
        >>> model = recognition_predictor(pretrained=True)
        >>> input_page = (255 * np.random.rand(32, 128, 3)).astype(np.uint8)
        >>> out = model([input_page])

    Args:
    ----
        arch: name of the architecture or model itself to use (e.g. 'crnn_vgg16_bn')
        pretrained: If True, returns a model pre-trained on our text recognition dataset
        **kwargs: optional parameters to be passed to the architecture

    Returns:
    -------
        Recognition predictor
    """
    return _predictor(arch, pretrained, **kwargs)