class PTQuantizerSpec(QuantizerSpec):
    _state_names = PTQSpecStateNames

    def __init__(
        self,
        num_bits: int,
        mode: QuantizationMode,
        signedness_to_force: Optional[bool],
        narrow_range: bool,
        half_range: bool,
        scale_shape: Tuple[int, ...],
        logarithm_scale: bool,
        is_quantized_on_export: bool = False,
        compression_lr_multiplier: float = None,
    ):
        """
        :param scale_shape: Shape of quantizer scale parameters
        :param logarithm_scale: Whether to use log of scale as optimized parameter instead of scale itself.
        :param compression_lr_multiplier: Used to increase/decrease gradients for quantization parameters.
        :param is_quantized_on_export: Export to onnx weights quantized or non quantized. Should not be True for
            activation quantizers.
        """
        super().__init__(num_bits, mode, signedness_to_force, narrow_range, half_range)
        self.per_channel = scale_shape != (1,)
        self.scale_shape = scale_shape
        self.logarithm_scale = logarithm_scale
        self.compression_lr_multiplier = compression_lr_multiplier
        self.is_quantized_on_export = is_quantized_on_export

    @classmethod
    def from_config(
        cls,
        qconfig: QuantizerConfig,
        narrow_range: bool,
        half_range: bool,
        scale_shape: Tuple[int],
        logarithm_scale: bool,
        is_quantized_on_export: bool,
        compression_lr_multiplier: float,
    ) -> "PTQuantizerSpec":
        return cls(
            qconfig.num_bits,
            qconfig.mode,
            qconfig.signedness_to_force,
            narrow_range,
            half_range,
            scale_shape,
            logarithm_scale,
            is_quantized_on_export,
            compression_lr_multiplier,
        )

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "PTQuantizationPoint":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        kwargs = {
            cls._state_names.NUM_BITS: state["num_bits"],
            cls._state_names.MODE: state["mode"],
            cls._state_names.SIGNED_TO_FORCE: state["signedness_to_force"],
            cls._state_names.NARROW_RANGE: state["narrow_range"],
            cls._state_names.HALF_RANGE: state["half_range"],
            cls._state_names.SCALE_SHAPE: state["scale_shape"],
            cls._state_names.LOGARITHM_SCALE: state["logarithm_scale"],
            cls._state_names.IS_QUANTIZED_ON_EXPORT: state["is_quantized_on_export"],
            cls._state_names.COMPRESSION_LR_MULTIPLIER: state["compression_lr_multiplier"],
        }
        return cls(**kwargs)

    def get_state(self):
        return {
            self._state_names.NUM_BITS: self.num_bits,
            self._state_names.MODE: self.mode,
            self._state_names.SIGNED_TO_FORCE: self.signedness_to_force,
            self._state_names.NARROW_RANGE: self.narrow_range,
            self._state_names.HALF_RANGE: self.half_range,
            self._state_names.SCALE_SHAPE: self.scale_shape,
            self._state_names.LOGARITHM_SCALE: self.logarithm_scale,
            self._state_names.IS_QUANTIZED_ON_EXPORT: self.is_quantized_on_export,
            self._state_names.COMPRESSION_LR_MULTIPLIER: self.compression_lr_multiplier,
        }

