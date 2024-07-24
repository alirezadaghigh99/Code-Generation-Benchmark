class FarSeg(Module):
    """Foreground-Aware Relation Network (FarSeg).

    This model can be used for binary- or multi-class object segmentation, such as
    building, road, ship, and airplane segmentation. It can be also extended as a change
    detection model. It features a foreground-scene relation module to model the
    relation between scene embedding, object context, and object feature, thus improving
    the discrimination of object feature representation.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/2011.09766.pdf
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        classes: int = 16,
        backbone_pretrained: bool = True,
    ) -> None:
        """Initialize a new FarSeg model.

        Args:
            backbone: name of ResNet backbone, one of ["resnet18", "resnet34",
                "resnet50", "resnet101"]
            classes: number of output segmentation classes
            backbone_pretrained: whether to use pretrained weight for backbone
        """
        super().__init__()
        if backbone in ['resnet18', 'resnet34']:
            max_channels = 512
        elif backbone in ['resnet50', 'resnet101']:
            max_channels = 2048
        else:
            raise ValueError(f'unknown backbone: {backbone}.')
        kwargs = {}
        if backbone_pretrained:
            kwargs = {
                'weights': getattr(
                    torchvision.models, f'ResNet{backbone[6:]}_Weights'
                ).DEFAULT
            }
        else:
            kwargs = {'weights': None}

        self.backbone = getattr(resnet, backbone)(**kwargs)

        self.fpn = FPN(
            in_channels_list=[max_channels // (2 ** (3 - i)) for i in range(4)],
            out_channels=256,
        )
        self.fsr = _FSRelation(max_channels, [256] * 4, 256)
        self.decoder = _LightWeightDecoder(256, 128, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: input image

        Returns:
            output prediction
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        features = [c2, c3, c4, c5]

        coarsest_features = features[-1]
        scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)
        fpn_features = self.fpn(
            OrderedDict({f'c{i + 2}': features[i] for i in range(4)})
        )
        features = [v for k, v in fpn_features.items()]
        features = self.fsr(scene_embedding, features)

        logit = self.decoder(features)

        return cast(Tensor, logit)

