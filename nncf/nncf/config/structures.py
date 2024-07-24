class QuantizationRangeInitArgs(NNCFExtraConfigStruct):
    """
    Stores additional arguments for quantization range initialization algorithms.

    :param data_loader: Provides an iterable over the given dataset.
    :param device: Device to perform initialization. If `device` is `None`
        then the device of the model parameters will be used.
    """

    def __init__(self, data_loader: NNCFDataLoader, device: Optional[str] = None):
        self._data_loader = data_loader
        self._device = device

    @property
    def data_loader(self) -> NNCFDataLoader:
        return self._data_loader

    @property
    def device(self) -> str:
        return self._device

    @classmethod
    def get_id(cls) -> str:
        return "quantization_range_init_args"

