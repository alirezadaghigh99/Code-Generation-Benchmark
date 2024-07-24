def from_bin(cls, binstring: str):
        """Creates a BitMask from a binary string."""
        return cls(_bin_to_int(binstring))

class BitMask:
    """Utility class to represent and manipulate tags.
    Attributes:
        x:
            An integer representation of the binary mask.
    Examples:
        >>> # the following are equivalent
        >>> mask = BitMask(6)
        >>> mask = BitMask.from_hex('0x6')
        >>> mask = Bitmask.from_bin('0b0110')
        >>> # for a dataset with 10 images, assume the following tag
        >>> # 0001011001 where the 1st, 4th, 5th and 7th image are selected
        >>> # this tag would be stored as 0x59.
        >>> hexstring = '0x59'                    # what you receive from the api
        >>> mask = BitMask.from_hex(hexstring)  # create a bitmask from it
        >>> indices = mask.to_indices()         # get list of indices which are one
        >>> # indices is [0, 3, 4, 6]
    """

    def __init__(self, x):
        self.x = x

    @classmethod
    def from_hex(cls, hexstring: str):
        """Creates a bit mask object from a hexstring."""
        return cls(_hex_to_int(hexstring))

    @classmethod
    def from_bin(cls, binstring: str):
        """Creates a BitMask from a binary string."""
        return cls(_bin_to_int(binstring))

    @classmethod
    def from_length(cls, length: int):
        """Creates a all-true bitmask of a predefined length"""
        binstring = "0b" + "1" * length
        return cls.from_bin(binstring)

    def to_hex(self):
        """Creates a BitMask from a hex string."""
        return _int_to_hex(self.x)

    def to_bin(self):
        """Returns a binary string representing the bit mask."""
        return _int_to_bin(self.x)

    def to_indices(self) -> List[int]:
        """Returns the list of indices bits which are set to 1 from the right.
        Examples:
            >>> mask = BitMask('0b0101')
            >>> indices = mask.to_indices()
            >>> # indices is [0, 2]
        """
        return _get_nonzero_bits(self.x)

    def invert(self, total_size: int):
        """Sets every 0 to 1 and every 1 to 0 in the bitstring.

        Args:
            total_size:
                Total size of the tag.

        """
        self.x = _invert(self.x, total_size)

    def complement(self):
        """Same as invert but with the appropriate name."""
        self.invert()

    def union(self, other):
        """Calculates the union of two bit masks.
        Examples:
            >>> mask1 = BitMask.from_bin('0b0011')
            >>> mask2 = BitMask.from_bin('0b1100')
            >>> mask1.union(mask2)
            >>> # mask1.binstring is '0b1111'
        """
        self.x = _union(self.x, other.x)

    def intersection(self, other):
        """Calculates the intersection of two bit masks.
        Examples:
            >>> mask1 = BitMask.from_bin('0b0011')
            >>> mask2 = BitMask.from_bin('0b1100')
            >>> mask1.intersection(mask2)
            >>> # mask1.binstring is '0b0000'
        """
        self.x = _intersection(self.x, other.x)

    def difference(self, other):
        """Calculates the difference of two bit masks.
        Examples:
            >>> mask1 = BitMask.from_bin('0b0111')
            >>> mask2 = BitMask.from_bin('0b1100')
            >>> mask1.difference(mask2)
            >>> # mask1.binstring is '0b0011'
        """
        self.union(other)
        self.x = self.x - other.x

    def __sub__(self, other):
        ret = copy.deepcopy(self)
        ret.difference(other)
        return ret

    def __eq__(self, other):
        return self.to_bin() == other.to_bin()

    def masked_select_from_list(self, list_: List):
        """Returns a subset of a list depending on the bitmask.

        The bitmask is read from right to left, i.e. the least significant bit
        corresponds to index 0.

        Examples:
            >>> list_to_subset = [4, 7, 9, 1]
            >>> mask = BitMask.from_bin("0b0101")
            >>> masked_list = mask.masked_select_from_list(list_to_subset)
            >>> # masked_list = [4, 9]

        """
        indices = self.to_indices()
        return [list_[index] for index in indices]

    def get_kth_bit(self, k: int) -> bool:
        """Returns the boolean value of the kth bit from the right."""
        return _get_kth_bit(self.x, k) > 0

    def set_kth_bit(self, k: int):
        """Sets the kth bit from the right to '1'.
        Examples:
            >>> mask = BitMask('0b0000')
            >>> mask.set_kth_bit(2)
            >>> # mask.binstring is '0b0100'
        """
        self.x = _set_kth_bit(self.x, k)

    def unset_kth_bit(self, k: int):
        """Unsets the kth bit from the right to '0'.
        Examples:
            >>> mask = BitMask('0b1111')
            >>> mask.unset_kth_bit(2)
            >>> # mask.binstring is '0b1011'
        """
        self.x = _unset_kth_bit(self.x, k)

