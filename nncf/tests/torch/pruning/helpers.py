def get_basic_pruning_config(input_sample_size=None) -> NNCFConfig:
    if input_sample_size is None:
        input_sample_size = [1, 1, 4, 4]
    config = NNCFConfig()
    config.update(
        {
            "model": "pruning_conv_model",
            "input_info": {
                "sample_size": input_sample_size,
            },
            "compression": {"params": {}},
        }
    )
    return config

class PruningTestModel(nn.Module):
    CONV_1_NODE_NAME = "PruningTestModel/NNCFConv2d[conv1]/conv2d_0"
    CONV_2_NODE_NAME = "PruningTestModel/NNCFConv2d[conv2]/conv2d_0"
    CONV_3_NODE_NAME = "PruningTestModel/NNCFConv2d[conv3]/conv2d_0"

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 3, 2, 9, -2)
        self.relu = nn.ReLU()
        self.conv2 = create_conv(3, 1, 3, -10, 0)
        self.conv3 = create_conv(1, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

