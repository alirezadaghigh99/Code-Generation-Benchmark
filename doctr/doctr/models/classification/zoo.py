def page_orientation_predictor(
    arch: str = "mobilenet_v3_small_page_orientation", pretrained: bool = False, **kwargs: Any
) -> OrientationPredictor:
    """Page orientation classification architecture.

    >>> import numpy as np
    >>> from doctr.models import page_orientation_predictor
    >>> model = page_orientation_predictor(arch='mobilenet_v3_small_page_orientation', pretrained=True)
    >>> input_page = (255 * np.random.rand(512, 512, 3)).astype(np.uint8)
    >>> out = model([input_page])

    Args:
    ----
        arch: name of the architecture to use (e.g. 'mobilenet_v3_small_page_orientation')
        pretrained: If True, returns a model pre-trained on our recognition crops dataset
        **kwargs: keyword arguments to be passed to the OrientationPredictor

    Returns:
    -------
        OrientationPredictor
    """
    return _orientation_predictor(arch, pretrained, **kwargs)