class FakeQuantize(FakeQuantizeBase):
    r"""Simulate the quantize and dequantize operations in training time.

    The output of this module is given by::

        x_out = (
          clamp(round(x/scale + zero_point), quant_min, quant_max) - zero_point
        ) * scale

    * :attr:`is_dynamic` indicates whether the fake quantie is a placeholder for dynamic quantization
      operators (choose_qparams -> q -> dq) or static quantization operators (q -> dq)

    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`fake_quant_enabled` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`observer_enabled` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
        allowable values are torch.qint8 and torch.quint8.

    Args:

        observer (module): Module for observing statistics on input tensors and calculating scale
          and zero-point.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        activation_post_process (Module): User provided module that collects statistics on the input tensor and
          provides a method to calculate scale and zero-point.

    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=None, quant_max=None, is_dynamic=False, **observer_kwargs):
        super().__init__()
        # Populate quant_min/quant_max to observer_kwargs if valid
        if quant_min is not None and quant_max is not None:
            assert quant_min <= quant_max, \
                'quant_min must be less than or equal to quant_max'
            dtype = observer_kwargs.get("dtype", torch.quint8)
            if hasattr(observer, "p"):
                # In case observer is _PartialWrapper, dtype can be stored in
                # observer.p.keywords["dtype"]
                dtype = getattr(getattr(observer, "p", {}), "keywords", {}).get(
                    "dtype", dtype
                )
            assert torch.iinfo(dtype).min <= quant_min, 'quant_min out of bound'
            assert quant_max <= torch.iinfo(dtype).max, 'quant_max out of bound'
            observer_kwargs.update({"quant_min": quant_min, "quant_max": quant_max})
        observer_kwargs["is_dynamic"] = is_dynamic
        self.activation_post_process = observer(**observer_kwargs)
        # TODO: keeping self.quant_min/max for BC; remove after a couple releases
        # Users should use self.activation_post_process.quant_min
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.is_dynamic = self.activation_post_process.is_dynamic
        if _is_float_qparams(self.activation_post_process.qscheme):
            zero_point_dtype = torch.float
        else:
            zero_point_dtype = torch.int
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=zero_point_dtype))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        assert _is_per_channel(self.qscheme) or \
            _is_per_tensor(self.qscheme), \
            'Only per channel and per tensor quantization are supported in fake quantize' + \
            ' got qscheme: ' + str(self.qscheme)
        self.is_per_channel = _is_per_channel(self.qscheme)

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                X = torch.fake_quantize_per_channel_affine(
                    X, self.scale, self.zero_point,
                    self.ch_axis, self.activation_post_process.quant_min, self.activation_post_process.quant_max)
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.activation_post_process.quant_min, self.activation_post_process.quant_max)
        return X

    @torch.jit.export
    def extra_repr(self):
        return f'fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, ' \
               f'quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, ' \
               f'dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, ' \
               f'scale={self.scale}, zero_point={self.zero_point}'

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == 'scale':
                    self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

class FusedMovingAvgObsFakeQuantize(FakeQuantize):
    r"""Define a fused module to observe the tensor.

    Fused module that is used to observe the input tensor (compute min/max), compute
    scale/zero_point and fake_quantize the tensor.
    This module uses calculation similar MovingAverageMinMaxObserver for the inputs,
    to compute the min/max values in order to compute the scale/zero_point.
    The qscheme input in the observer is used to differentiate between symmetric/affine
    quantization scheme.

    The output of this module is given by
    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale

    Similar to :class:`~torch.ao.quantization.FakeQuantize`, and accepts the same attributes as the
    base class.

    """

    def __init__(
        self,
        observer: Any = MovingAverageMinMaxObserver,
        quant_min: int = 0,
        quant_max: int = 255,
        **observer_kwargs: Any
    ) -> None:
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        assert isinstance(self.activation_post_process, (MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver)), \
            "Fused observer+fake_quant module only works with MovingAverageMinMaxObserver"
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.long))
        self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.long))
        self.is_symmetric_quant = _is_symmetric_quant(self.activation_post_process.qscheme)

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.activation_post_process.calculate_qparams()

    @torch.jit.export
    def extra_repr(self) -> str:
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"scale={self.scale}, zero_point={self.zero_point}, dtype={self.dtype}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"qscheme={self.qscheme}, reduce_range={self.activation_post_process.reduce_range}"
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.fused_moving_avg_obs_fake_quant(
            X,
            self.observer_enabled,
            self.fake_quant_enabled,
            self.activation_post_process.min_val,
            self.activation_post_process.max_val,
            self.scale,
            self.zero_point,
            self.activation_post_process.averaging_constant,
            self.activation_post_process.quant_min,
            self.activation_post_process.quant_max,
            self.ch_axis,
            self.is_per_channel,
            self.is_symmetric_quant,
        )

