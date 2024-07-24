def _min_adj(bits, low, range_len, narrow_range):
        quants_count = 2**bits - (2 if narrow_range else 1)
        return range_len / quants_count * tf.round(quants_count * low / range_len)

class TFQuantizerSpec(QuantizerSpec):
    def __init__(
        self,
        num_bits: int,
        mode: QuantizationMode,
        signedness_to_force: Optional[bool],
        narrow_range: bool,
        half_range: bool,
        per_channel: bool,
    ):
        super().__init__(num_bits, mode, signedness_to_force, narrow_range, half_range)
        self.per_channel = per_channel

    @classmethod
    def from_config(cls, qconfig: QuantizerConfig, narrow_range: bool, half_range: bool) -> "TFQuantizerSpec":
        return cls(
            qconfig.num_bits, qconfig.mode, qconfig.signedness_to_force, narrow_range, half_range, qconfig.per_channel
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
            "narrow_range": self.narrow_range,
            "half_range": self.half_range,
            "per_channel": self.per_channel,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TFQuantizerSpec":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)

