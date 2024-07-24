class QuantizedLinear(QuantizedLayer, nn.Linear):
    """Linear layer with quantization aware training capability"""

    CONFIG_ATTRIBUTES = QuantizedLayer.CONFIG_ATTRIBUTES + [
        "activation_bits",
        "requantize_output",
        "ema_decay",
    ]
    REPR_ATTRIBUTES = QuantizedLayer.REPR_ATTRIBUTES + [
        "activation_bits",
        "accumulation_bits",
        "ema_decay",
        "requantize_output",
    ]

    def __init__(
        self, *args, activation_bits=8, requantize_output=True, ema_decay=0.9999, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if activation_bits < 2:
            raise ValueError(f"activation_bits={activation_bits} must be higher than 1 ")
        self.activation_bits = activation_bits
        self.accumulation_bits = 32
        self.ema_decay = ema_decay
        self.requantize_output = requantize_output
        self.register_buffer("input_thresh", torch.zeros(1))
        if self.requantize_output:
            self.register_buffer("output_thresh", torch.zeros(1))
        # real quantization
        if kwargs.get("bias", True):
            self.register_buffer("_quantized_bias", None)
            self.register_buffer("bias_scale", None)

    def training_quantized_forward(self, input):
        """fake quantized forward, fake quantizes weights and activations,
        learn quantization ranges if quantization mode is EMA.
        This function should only be used while training"""
        assert self.training, "should only be called when training"
        if self.mode == QuantizationMode.EMA:
            self._update_ema(self.input_thresh, input.detach())
        input_scale = self._get_input_scale(input)
        out = F.linear(
            _fake_quantize(input, input_scale, self.activation_bits),
            self.fake_quantized_weight,
            self.bias,
        )
        if self.requantize_output:
            if self.mode == QuantizationMode.EMA:
                self._update_ema(self.output_thresh, out.detach())
            out = _fake_quantize(out, self._get_output_scale(out), self.activation_bits)
        return out

    def inference_quantized_forward(self, input):
        """Simulate quantized inference. quantize input and perform calculation with only integer numbers.
        This function should only be used while doing inference"""
        assert not self.training, "should only be called when not training"
        input_scale = self._get_input_scale(input)
        self.bias_scale = self.weight_scale * input_scale
        quantized_input = quantize(input, input_scale, self.activation_bits)
        out = F.linear(quantized_input, self.quantized_weight, self.quantized_bias)
        # TODO(ofir) fuse the operation of requantization with dequantiz
        out = dequantize(out, self.bias_scale)
        if self.requantize_output:
            output_scale = self._get_output_scale(out)
            out = dequantize(quantize(out, output_scale, self.activation_bits), output_scale)
        return out

    def _eval(self):
        super()._eval()
        if self.mode == QuantizationMode.EMA and self.bias is not None:
            self.bias_scale = self._get_input_scale() * self.weight_scale
            self.quantized_bias = quantize(self.bias, self.bias_scale, self.accumulation_bits)

    @staticmethod
    def _state_dict_hook(module, state_dict, prefix, local_metadata):
        """hook to be registered to module when exporting the model to 8bit,\
             can be overrided to customize to layer behaviour"""
        super()._state_dict_hook(module, state_dict, prefix, local_metadata)
        if module.mode_8bit:
            if module.mode == QuantizationMode.EMA:
                state_dict.pop(prefix + "bias", None)
                try:
                    state_dict[prefix + "_quantized_bias"] = state_dict[
                        prefix + "_quantized_bias"
                    ].int()
                except KeyError:
                    # in case there is no bias dont do anything
                    pass
        else:
            state_dict.pop(prefix + "_quantized_bias", None)
            state_dict.pop(prefix + "bias_scale", None)

    @property
    def quantized_bias(self):
        try:
            if self.mode == QuantizationMode.EMA:
                bias = self._quantized_bias
            elif self.mode == QuantizationMode.DYNAMIC:
                bias = quantize(self.bias, self.bias_scale, self.accumulation_bits)
            else:
                raise RuntimeError(f"Unknown quantization mode: {self.mode}")
        except AttributeError:
            bias = None
        return bias

    @quantized_bias.setter
    def quantized_bias(self, value):
        self._quantized_bias = value

    def _get_input_scale(self, input=None):
        return self._get_activation_scale(input, self.input_thresh)

    def _get_output_scale(self, output=None):
        return self._get_activation_scale(output, self.output_thresh)

    def _get_activation_scale(self, activation, threshold):
        if self.mode == QuantizationMode.DYNAMIC:
            scale = get_dynamic_scale(activation, self.activation_bits)
        elif self.mode == QuantizationMode.EMA:
            scale = get_scale(self.activation_bits, threshold)
        return scale

    def _update_ema(self, ema, input, reduce_fn=lambda x: x.abs().max()):
        """Update exponential moving average (EMA) of activations thresholds.
        the reduce_fn calculates the current threshold from the input tensor"""
        assert self._step >= self.start_step
        if self._step == self.start_step:
            ema.fill_(reduce_fn(input))
        else:
            ema.sub_((1 - self.ema_decay) * (ema - reduce_fn(input)))

class QuantizedEmbedding(QuantizedLayer, nn.Embedding):
    """Embedding layer with quantization aware training capability"""

    def training_quantized_forward(self, input):
        """Return quantized embeddings"""
        assert self.training, "should only be called when training"
        return F.embedding(
            input,
            self.fake_quantized_weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def inference_quantized_forward(self, input):
        """forward to be used during inference"""
        assert not self.training, "should only be called when not training"
        q_embeddings = F.embedding(
            input,
            self.quantized_weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return dequantize(q_embeddings, self.weight_scale)

class FakeLinearQuantizationWithSTE(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop"""

    @staticmethod
    def forward(ctx, input, scale, bits=8):
        """fake quantize input according to scale and number of bits, dequantize
        quantize(input))"""
        return dequantize(quantize(input, scale, bits), scale)

    @staticmethod
    def backward(ctx, grad_output):
        """Calculate estimated gradients for fake quantization using
        Straight-Through Estimator (STE) according to:
        https://openreview.net/pdf?id=B1ae1lZRb"""
        return grad_output, None, None

