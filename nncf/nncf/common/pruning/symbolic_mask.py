class SymbolicMask(NNCFTensor):
    """
    Framework agnostic 1D NNCFTensor representation which only uses given dimension and do not uses value
    of the tensor. Keeps additional attribute - symbolic mask producer, pointer to NNCFNode which produced
    this mask during symbolic mask propagation algorithm. NNCFNode produced a (symbolic or not) mask means
    this mask was set as an output mask to this NNCFNode during (symbolic or not) mask propagation.
    Tensor shape and mask producer attributes are correctly propagating during
    symbolic mask propagation by SymbolicMaskProcessor.
    """

    def __init__(self, dimension: int, mask_producers: Union[int, List[SymbolicMaskProducer]] = None):
        super().__init__(None)
        self._mask_producers = mask_producers
        if mask_producers is None:
            self._mask_producers = []
        elif isinstance(mask_producers, int):
            self._mask_producers = [SymbolicMaskProducer(mask_producers)]

        self._shape = dimension

    @property
    def shape(self) -> List[int]:
        return [self._shape]

    @property
    def mask_producers(self) -> List[SymbolicMaskProducer]:
        return self._mask_producers

    @property
    def device(self) -> None:
        return None

