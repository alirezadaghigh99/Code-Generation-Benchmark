def crop_orientation_predictor(
    arch: str = "mobilenet_v3_small_crop_orientation", pretrained: bool = False, **kwargs: Any
) -> OrientationPredictor:
    """Crop orientation classification architecture.

    >>> import numpy as np
    >>> from doctr.models import crop_orientation_predictor
    >>> model = crop_orientation_predictor(arch='mobilenet_v3_small_crop_orientation', pretrained=True)
    >>> input_crop = (255 * np.random.rand(256, 256, 3)).astype(np.uint8)
    >>> out = model([input_crop])

    Args:
    ----
        arch: name of the architecture to use (e.g. 'mobilenet_v3_small_crop_orientation')
        pretrained: If True, returns a model pre-trained on our recognition crops dataset
        **kwargs: keyword arguments to be passed to the OrientationPredictor

    Returns:
    -------
        OrientationPredictor
    """
    return _orientation_predictor(arch, pretrained, **kwargs)

