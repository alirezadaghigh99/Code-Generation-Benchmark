def yolov5_mobilenet_v3_small_fpn(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs,
):
    """
    Constructs a high resolution YOLOv5 model with a MobileNetV3-Large FPN backbone.
    It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
    :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn` for more
    details.

    Note:
        We do not provide a pre-trained model with mobilenet as the backbone now, this function
        is just used as an example of how to construct a YOLOv5 model with TorchVision's pre-trained
        MobileNetV3-Small FPN backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting
            from final block. Valid values are between 0 and 6, with 6 meaning all backbone layers
            are trainable.
    """
    weights_name = "yolov5_mobilenet_v3_small_fpn_coco"

    return _yolov5_mobilenet_v3_small_fpn(
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs,
    )