class QuantizationSpec(QuantizationSpecBase):
    """Quantization spec for common operators that allows user to specify how to
    quantize a Tensor, this includes dtype, quant_min, quant_max etc.
    """

    dtype: torch.dtype
    # observer or fake_quantize constructor such as
    # MinMaxObserver, PerChannelHistogramObserver etc.
    # or we can attach some custom args to them
    # e.g. MinMaxObserver.with_args(eps=eps)
    observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    qscheme: Optional[torch.qscheme] = None
    ch_axis: Optional[int] = None
    is_dynamic: bool = False

    def __post_init__(self):
        # TODO: add init for quant_min/quant_max
        # quant_min must be less than quant_max
        if (
            self.quant_min is not None
            and self.quant_max is not None
            and self.quant_min > self.quant_max
        ):
            raise ValueError(
                f"quant_min {self.quant_min} must be <= quant_max {self.quant_max}."
            )

        # ch_axis must be less than the number of channels
        # but no way to check here. Just check that it is not < 0.
        if self.ch_axis is not None and self.ch_axis < 0:
            raise ValueError("Ch_axis is < 0.")

