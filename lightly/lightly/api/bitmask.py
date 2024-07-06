def from_bin(cls, binstring: str):
        """Creates a BitMask from a binary string."""
        return cls(_bin_to_int(binstring))

