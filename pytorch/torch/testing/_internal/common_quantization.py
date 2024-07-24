def get_supported_device_types():
    return ['cpu', 'cuda'] if torch.cuda.is_available() and not TEST_WITH_ROCM else ['cpu']

def _dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # source: https://github.com/pytorch-labs/gpt-fast/blob/main/quantize.py
    # default setup for affine quantization of activations
    x_dtype = x.dtype
    x = x.float()
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales.to(x_dtype), zero_points

class LSTMwithHiddenDynamicModel(torch.nn.Module):
    def __init__(self, qengine='fbgemm'):
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.lstm = torch.nn.LSTM(2, 2).to(dtype=torch.float)

    def forward(self, x, hid):
        x, hid = self.lstm(x, hid)
        return x, hid

class TwoLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)

