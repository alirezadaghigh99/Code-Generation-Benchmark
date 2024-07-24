def create_compressed_model_and_algo_for_test(
    model: Module,
    config: NNCFConfig = None,
    dummy_forward_fn: Callable[[Module], Any] = None,
    wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
    compression_state: Dict[str, Any] = None,
) -> Tuple[NNCFNetwork, PTCompressionAlgorithmController]:
    if config is not None:
        assert isinstance(config, NNCFConfig)
        NNCFConfig.validate(config)
    algo, model = create_compressed_model(
        model,
        config,
        dump_graphs=False,
        dummy_forward_fn=dummy_forward_fn,
        wrap_inputs_fn=wrap_inputs_fn,
        compression_state=compression_state,
    )
    return model, algo

def set_torch_seed(seed: int = 42):
    saved_seed = torch.seed()
    torch.manual_seed(seed)
    yield
    torch.manual_seed(saved_seed)

def get_empty_config(
    model_size=4, input_sample_sizes: Union[Tuple[List[int]], List[int]] = None, input_info: Dict = None
) -> NNCFConfig:
    if input_sample_sizes is None:
        input_sample_sizes = [1, 1, 4, 4]

    def _create_input_info():
        if isinstance(input_sample_sizes, tuple):
            return [{"sample_size": sizes} for sizes in input_sample_sizes]
        return [{"sample_size": input_sample_sizes}]

    config = NNCFConfig()
    config.update(
        {
            "model": "empty_config",
            "model_size": model_size,
            "input_info": input_info if input_info else _create_input_info(),
        }
    )
    return config

class TwoConvTestModel(nn.Module):
    INPUT_SHAPE = [1, 1, 4, 4]
    NNCF_CONV_NODES_NAMES = [
        "TwoConvTestModel/Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0",
        "TwoConvTestModel/Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0",
    ]
    CONV_NODES_NAMES = [
        "TwoConvTestModel/Sequential[features]/Sequential[0]/Conv2d[0]/conv2d_0",
        "TwoConvTestModel/Sequential[features]/Sequential[1]/Conv2d[0]/conv2d_0",
    ]

    def __init__(self):
        super().__init__()
        self.features = []
        self.features.append(nn.Sequential(create_conv(1, 2, 2, -1, -2)))
        self.features.append(nn.Sequential(create_conv(2, 1, 3, 0, 0)))
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)

    @property
    def weights_num(self):
        return 8 + 18

    @property
    def bias_num(self):
        return 2 + 1

    @property
    def nz_weights_num(self):
        return 4 + 6

    @property
    def nz_bias_num(self):
        return 2

class BasicConvTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self, in_channels=1, out_channels=2, kernel_size=2, weight_init=-1, bias_init=-2, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.conv = create_conv(in_channels, out_channels, kernel_size, weight_init, bias_init, padding)
        self.wq_scale_shape_per_channel = (out_channels, 1, 1, 1)
        self.aq_scale_shape_per_channel = (1, in_channels, 1, 1)

    @staticmethod
    def default_weight():
        return torch.tensor([[[[0.0, -1.0], [-1.0, 0.0]]], [[[0.0, -1.0], [-1.0, 0.0]]]])

    @staticmethod
    def default_bias():
        return torch.tensor([-2.0, -2])

    def forward(self, x):
        return self.conv(x)

    @property
    def weights_num(self):
        return self.out_channels * self.kernel_size**2

    @property
    def bias_num(self):
        return self.kernel_size

    @property
    def nz_weights_num(self):
        return self.kernel_size * self.out_channels

    @property
    def nz_bias_num(self):
        return self.kernel_size

class Command(BaseCommand):
    def run(self, timeout=3600, assert_returncode_zero=True):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # See runs_subprocess_in_precommit for more info on why this is needed
        return super().run(timeout, assert_returncode_zero)

class EmptyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input_, **kwargs):
        return None

class RandomDatasetMock(BaseDatasetMock):
    def __getitem__(self, index):
        return torch.rand(self._input_size), torch.zeros(1)

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.field = nn.Linear(1, 1)

    def forward(self, *input_, **kwargs):
        return None

