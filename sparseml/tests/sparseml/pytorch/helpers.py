def create_optim_sgd(model: Module, lr: float = 0.0001) -> SGD:
    return SGD(model.parameters(), lr=lr)

class MLPNet(Module):
    _LAYER_DESCS = None

    @staticmethod
    def layer_descs() -> List[LayerDesc]:
        if MLPNet._LAYER_DESCS is None:
            MLPNet._LAYER_DESCS = []
            model = MLPNet()

            for name, layer in model.named_modules():
                if isinstance(layer, Linear):
                    MLPNet._LAYER_DESCS.append(
                        LayerDesc(
                            name,
                            [layer.in_features],
                            [layer.out_features],
                            layer.bias is not None,
                        )
                    )
                elif isinstance(layer, ReLU):
                    MLPNet._LAYER_DESCS.append(
                        LayerDesc(
                            name,
                            [],
                            [],
                            False,
                        )
                    )

        return MLPNet._LAYER_DESCS

    def __init__(self):
        super().__init__()
        self.seq = Sequential(
            OrderedDict(
                [
                    ("fc1", Linear(8, 16, bias=True)),
                    ("act1", ReLU()),
                    ("fc2", Linear(16, 32, bias=True)),
                    ("act2", ReLU()),
                    ("fc3", Linear(32, 64, bias=True)),
                    ("sig", Sigmoid()),
                ]
            )
        )

    def forward(self, inp: Tensor):
        return self.seq(inp)

class ConvNet(Module):
    _LAYER_DESCS = None

    @staticmethod
    def layer_descs() -> List[LayerDesc]:
        if ConvNet._LAYER_DESCS is None:
            ConvNet._LAYER_DESCS = [
                LayerDesc("seq.conv1", (3, 28, 28), (16, 14, 14), True),
                LayerDesc("seq.act1", [16, 14, 14], (16, 14, 14), False),
                LayerDesc("seq.conv2", (16, 14, 14), (32, 7, 7), True),
                LayerDesc("seq.act2", (32, 7, 7), (32, 7, 7), False),
                LayerDesc("mlp.fc", (32,), (10,), True),
            ]

        return ConvNet._LAYER_DESCS

    def __init__(self):
        super().__init__()
        self.seq = Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=True),
                    ),
                    ("act1", ReLU()),
                    (
                        "conv2",
                        Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True),
                    ),
                    ("act2", ReLU()),
                ]
            )
        )
        self.pool = AdaptiveAvgPool2d(1)
        self.mlp = Sequential(
            OrderedDict([("fc", Linear(32, 10, bias=True)), ("sig", Sigmoid())])
        )

    def forward(self, inp: Tensor):
        out = self.seq(inp)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        classes = self.mlp(out)

        return classes

class LinearNet(Module):
    _LAYER_DESCS = None

    @staticmethod
    def layer_descs() -> List[LayerDesc]:
        if LinearNet._LAYER_DESCS is None:
            LinearNet._LAYER_DESCS = []
            model = LinearNet()

            for name, layer in model.named_modules():
                if not isinstance(layer, Linear):
                    continue

                LinearNet._LAYER_DESCS.append(
                    LayerDesc(
                        name,
                        [layer.in_features],
                        [layer.out_features],
                        layer.bias is not None,
                    )
                )

        return LinearNet._LAYER_DESCS

    def __init__(self):
        super().__init__()
        self.seq = Sequential(
            OrderedDict(
                [
                    ("fc1", Linear(8, 16, bias=True)),
                    ("fc2", Linear(16, 32, bias=True)),
                    (
                        "block1",
                        Sequential(
                            OrderedDict(
                                [
                                    ("fc1", Linear(32, 16, bias=True)),
                                    ("fc2", Linear(16, 8, bias=True)),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, inp: Tensor):
        return self.seq(inp)

