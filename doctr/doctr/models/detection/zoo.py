def detection_predictor(
    arch: Any = "fast_base",
    pretrained: bool = False,
    assume_straight_pages: bool = True,
    **kwargs: Any,
) -> DetectionPredictor:
    """Text detection architecture.

    >>> import numpy as np
    >>> from doctr.models import detection_predictor
    >>> model = detection_predictor(arch='db_resnet50', pretrained=True)
    >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([input_page])

    Args:
    ----
        arch: name of the architecture or model itself to use (e.g. 'db_resnet50')
        pretrained: If True, returns a model pre-trained on our text detection dataset
        assume_straight_pages: If True, fit straight boxes to the page
        **kwargs: optional keyword arguments passed to the architecture

    Returns:
    -------
        Detection predictor
    """
    return _predictor(arch, pretrained, assume_straight_pages, **kwargs)