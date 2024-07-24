class DeQuantize(torch.nn.Module):
    r"""Dequantizes an incoming tensor

    Examples::
        >>> input = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> # xdoctest: +SKIP
        >>> quantized_input = qm(input)
        >>> dqm = DeQuantize()
        >>> dequantized = dqm(quantized_input)
        >>> print(dequantized)
        tensor([[ 1., -1.],
                [ 1., -1.]], dtype=torch.float32)
    """

    def forward(self, Xq):
        return Xq.dequantize()

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        return DeQuantize()

class Quantize(torch.nn.Module):
    r"""Quantizes an incoming tensor

    Args:
     `scale`: scale of the output Quantized Tensor
     `zero_point`: zero_point of output Quantized Tensor
     `dtype`: data type of output Quantized Tensor
     `factory_kwargs`: Dictionary of kwargs used for configuring initialization
         of internal buffers. Currently, `device` and `dtype` are supported.
         Example: `factory_kwargs={'device': 'cuda', 'dtype': torch.float64}`
         will initialize internal buffers as type `torch.float64` on the current CUDA device.
         Note that `dtype` only applies to floating-point buffers.

    Examples::
        >>> t = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> # xdoctest: +SKIP
        >>> qt = qm(t)
        >>> print(qt)
        tensor([[ 1., -1.],
                [ 1., -1.]], size=(2, 2), dtype=torch.qint8, scale=1.0, zero_point=2)
    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, scale, zero_point, dtype, factory_kwargs=None):
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        super().__init__()
        self.register_buffer('scale', torch.tensor([scale], **factory_kwargs))
        self.register_buffer('zero_point',
                             torch.tensor([zero_point], dtype=torch.long,
                                          **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        self.dtype = dtype

    def forward(self, X):
        return torch.quantize_per_tensor(X, float(self.scale),
                                         int(self.zero_point), self.dtype)

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        assert hasattr(mod, 'activation_post_process')
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Quantize(scale.float().item(), zero_point.long().item(), mod.activation_post_process.dtype)

    def extra_repr(self):
        return f'scale={self.scale}, zero_point={self.zero_point}, dtype={self.dtype}'

