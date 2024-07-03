class QCommsConfig:
    """
    Quantization configs for the AllToAll and ReduceScatter communication modules used in sharding.
    """

    # Quantization of comm modules in the forward pass
    forward_precision: CommType = CommType.FP32
    # Quantization of comm modules in the backward pass
    backward_precision: CommType = CommType.FP32
    # For supported precisions (currently FP16), scale the gradient of the decoder and
    # divide the gradient of the encoder by this value. In some cases this can provide additional numerical stability.
    forward_loss_scale: Optional[float] = None
    backward_loss_scale: Optional[float] = None
    fp8_quantize_dim: Optional[int] = None
    fp8_quantize_dim_bwd: Optional[int] = None
    fp8_bwd_uses_143: Optional[bool] = False

    def __post_init__(self) -> None:
        if (
            self.forward_precision != CommType.FP8
            and self.backward_precision != CommType.FP8
            and (
                self.fp8_quantize_dim is not None
                or self.fp8_quantize_dim_bwd is not None
            )
        ):
            raise ValueError(
                f"fp8_quantize_dim is set to {self.fp8_quantize_dim} and fp8_quantize_dim_bwd is set to {self.fp8_quantize_dim_bwd} but no FP8 precision is found in forward or backward precisions"
            )
        if (
            self.backward_precision == CommType.FP8
            and self.fp8_quantize_dim_bwd is None
        ):
            self.fp8_quantize_dim_bwd = self.fp8_quantize_dim
            logger.warning(
                f"No override of FP8 bwd row dim, using general FP8 row dim for backward: {self.fp8_quantize_dim_bwd} "
            )