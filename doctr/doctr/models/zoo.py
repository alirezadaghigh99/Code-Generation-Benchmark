def ocr_predictor(
    det_arch: Any = "fast_base",
    reco_arch: Any = "crnn_vgg16_bn",
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    assume_straight_pages: bool = True,
    preserve_aspect_ratio: bool = True,
    symmetric_pad: bool = True,
    export_as_straight_boxes: bool = False,
    detect_orientation: bool = False,
    straighten_pages: bool = False,
    detect_language: bool = False,
    **kwargs: Any,
) -> OCRPredictor:
    """End-to-end OCR architecture using one model for localization, and another for text recognition.

    >>> import numpy as np
    >>> from doctr.models import ocr_predictor
    >>> model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)
    >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([input_page])

    Args:
    ----
        det_arch: name of the detection architecture or the model itself to use
            (e.g. 'db_resnet50', 'db_mobilenet_v3_large')
        reco_arch: name of the recognition architecture or the model itself to use
            (e.g. 'crnn_vgg16_bn', 'sar_resnet31')
        pretrained: If True, returns a model pre-trained on our OCR dataset
        pretrained_backbone: If True, returns a model with a pretrained backbone
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        preserve_aspect_ratio: If True, pad the input document image to preserve the aspect ratio before
            running the detection model on it.
        symmetric_pad: if True, pad the image symmetrically instead of padding at the bottom-right.
        export_as_straight_boxes: when assume_straight_pages is set to False, export final predictions
            (potentially rotated) as straight bounding boxes.
        detect_orientation: if True, the estimated general page orientation will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        straighten_pages: if True, estimates the page general orientation
            based on the segmentation map median line orientation.
            Then, rotates page before passing it again to the deep learning detection module.
            Doing so will improve performances for documents with page-uniform rotations.
        detect_language: if True, the language prediction will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        kwargs: keyword args of `OCRPredictor`

    Returns:
    -------
        OCR predictor
    """
    return _predictor(
        det_arch,
        reco_arch,
        pretrained,
        pretrained_backbone=pretrained_backbone,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        export_as_straight_boxes=export_as_straight_boxes,
        detect_orientation=detect_orientation,
        straighten_pages=straighten_pages,
        detect_language=detect_language,
        **kwargs,
    )