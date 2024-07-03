class ChangeStarFarSeg(ChangeStar):
    """The network architecture of ChangeStar(FarSeg).

    ChangeStar(FarSeg) is composed of a FarSeg model and a ChangeMixin module.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2108.07002
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        classes: int = 1,
        backbone_pretrained: bool = True,
    ) -> None:
        """Initializes a new ChangeStarFarSeg model.

        Args:
            backbone: name of ResNet backbone
            classes: number of output segmentation classes
            backbone_pretrained: whether to use pretrained weight for backbone
        """
        model = FarSeg(
            backbone=backbone, classes=classes, backbone_pretrained=backbone_pretrained
        )
        seg_classifier: Module = model.decoder.classifier
        model.decoder.classifier = nn.modules.Identity()  # type: ignore[assignment]

        super().__init__(
            dense_feature_extractor=model,
            seg_classifier=seg_classifier,
            changemixin=ChangeMixin(
                in_channels=128 * 2, inner_channels=16, num_convs=4, scale_factor=4.0
            ),
            inference_mode='t1t2',
        )