def from_config(cls, qconfig: QuantizerConfig, narrow_range: bool, half_range: bool) -> "QuantizerSpec":
        return cls(qconfig.num_bits, qconfig.mode, qconfig.signedness_to_force, narrow_range, half_range)

def from_config(cls, qconfig: QuantizerConfig, narrow_range: bool, half_range: bool) -> "QuantizerSpec":
        return cls(qconfig.num_bits, qconfig.mode, qconfig.signedness_to_force, narrow_range, half_range)

def from_config(cls, qconfig: QuantizerConfig, narrow_range: bool, half_range: bool) -> "QuantizerSpec":
        return cls(qconfig.num_bits, qconfig.mode, qconfig.signedness_to_force, narrow_range, half_range)

def from_config(cls, qconfig: QuantizerConfig, narrow_range: bool, half_range: bool) -> "QuantizerSpec":
        return cls(qconfig.num_bits, qconfig.mode, qconfig.signedness_to_force, narrow_range, half_range)

class QuantizerConfig:
    """
    A generic, framework-agnostic information on a configuration of a quantizer for abstract reasoning
    and determination of a quantizer setup scheme for a given model.
    """

    def __init__(
        self,
        num_bits: int = QUANTIZATION_BITS,
        mode: QuantizationScheme = QuantizationScheme.SYMMETRIC,
        signedness_to_force: Optional[bool] = None,
        per_channel: bool = QUANTIZATION_PER_CHANNEL,
    ):
        """
        :param num_bits: Bitwidth of the quantization.
        :param mode: The mode of quantization (symmetric or asymmetric).
        :param signedness_to_force: True if the quantizer *must* be signed, False if *must* be unsigned,
            None if the signed/unsigned attribute should be determined based on the incoming activation
            statistics during range initialization.
        :param per_channel: True for per-channel quantization, False for per-tensor.
        """
        self.num_bits = num_bits
        self.mode = mode
        self.signedness_to_force = signedness_to_force
        self.per_channel = per_channel

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return "B:{bits} M:{mode} SGN:{signedness} PC:{per_channel}".format(
            bits=self.num_bits,
            mode="S" if self.mode == QuantizationScheme.SYMMETRIC else "A",
            signedness="ANY" if self.signedness_to_force is None else ("S" if self.signedness_to_force else "U"),
            per_channel="Y" if self.per_channel else "N",
        )

    def __hash__(self):
        return hash(str(self))

    def is_valid_requantization_for(self, other: "QuantizerConfig") -> bool:
        """
        Quantizer config A is a valid requantization for quantizer config B if A is more strict -
        specifically, it might be reasonable to put quantizer A after quantizer B in tensor data control flow, so that
        the requantization will further constrain the input tensor data w.r.t. values it can take, but
        putting quantizer A after quantizer B would be unreasonable.

        :param other: The "primary" QuantizerConfig, i.e. the one that defines an already present quantization.
        :return: True if the current config is a valid requantization for `other`, False otherwise.
        """
        fail_conditions = [
            self.num_bits > other.num_bits,
            self.mode is QuantizationScheme.ASYMMETRIC and other.mode is QuantizationScheme.SYMMETRIC,
            self.signedness_to_force is None and other.signedness_to_force is not None,
            self.signedness_to_force is True and other.signedness_to_force is False,
        ]
        if any(fail_conditions):
            return False
        return True

    def compatible_with_a_unified_scale_linked_qconfig(self, linked_qconfig: "QuantizerConfig"):
        """
        For two configs to be compatible in a unified scale scenario, all of their fundamental parameters
        must be aligned.

        :param linked_qconfig: A QuantizerConfig that is compared against the current config.
        :return: A boolean value specifying whether `linked_qconfig` is compatible with the current config in terms
            of scale unification.
        """
        return (
            self.num_bits == linked_qconfig.num_bits
            and self.mode == linked_qconfig.mode
            and self.signedness_to_force == linked_qconfig.signedness_to_force
            and self.per_channel == linked_qconfig.per_channel
        )

    def is_a_bitwidth_variant(self, other_qconfig: "QuantizerConfig") -> bool:
        """
        :param other_qconfig: A QuantizerConfig to be compared against the current config.
        :return: A boolean value specifying whether `other_config` is identical to the current config
            in everything except the bitwidth.
        """
        return (
            self.per_channel == other_qconfig.per_channel
            and self.signedness_to_force == other_qconfig.signedness_to_force
            and self.mode == other_qconfig.mode
        )

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            "num_bits": self.num_bits,
            "mode": self.mode,
            "signedness_to_force": self.signedness_to_force,
            "per_channel": self.per_channel,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "QuantizerConfig":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)

class NonWeightQuantizerId(QuantizerId):
    """
    Unique identifier of a quantizer, which corresponds to non-weight operations, such as
    ordinary activation, function and input
    """

    def __init__(self, target_node_name: NNCFNodeName, input_port_id=None):
        self.target_node_name = target_node_name
        self.input_port_id = input_port_id

    def get_base(self) -> str:
        return self.target_node_name

    def get_suffix(self) -> str:
        return "|OUTPUT" if self.input_port_id is None else "|INPUT{}".format(self.input_port_id)

