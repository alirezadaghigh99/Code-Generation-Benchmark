class Dataset(Generic[DataItem, ModelInput]):
    """
    Wrapper for passing custom user datasets into NNCF algorithms.

    This class defines the interface by which compression algorithms
    retrieve data items from the passed data source object. These data items are used
    for different purposes, for example, model inference and model validation, based
    on the choice of the exact compression algorithm.

    If the data item has been returned from the data source per iteration and it cannot be
    used as input for model inference, the transformation function is used to extract the
    model's input from this data item. For example, in supervised learning, the data item
    usually contains both examples and labels. So transformation function should extract
    the examples from the data item.

    :param data_source: The iterable object serving as the source of data items.
    :param transform_func: The function that is used to extract the model's input
        from the data item. The data item here is the data item that is returned from
        the data source per iteration. This function should be passed when
        the data item cannot be directly used as model's input. If this is not specified, then the data item
        will be passed into the model as-is.
    """

    def __init__(
        self, data_source: Iterable[DataItem], transform_func: Optional[Callable[[DataItem], ModelInput]] = None
    ):
        self._data_source = data_source
        self._transform_func = transform_func

    def get_data(self, indices: Optional[List[int]] = None) -> Iterable[DataItem]:
        """
        Returns the iterable object that contains selected data items from the data source as-is.

        :param indices: The zero-based indices of data items that should be selected from
            the data source. The indices should be sorted in ascending order. If indices are
            not passed all data items are selected from the data source.
        :return: The iterable object that contains selected data items from the data source as-is.
        """
        return DataProvider(self._data_source, None, indices)

    def get_inference_data(self, indices: Optional[List[int]] = None) -> Iterable[ModelInput]:
        """
        Returns the iterable object that contains selected data items from the data source, for which
        the transformation function was applied. The item, which was returned per iteration from this
        iterable, can be used as the model's input for model inference.

        :param indices: The zero-based indices of data items that should be selected from
            the data source. The indices should be sorted in ascending order. If indices are
            not passed all data items are selected from the data source.
        :return: The iterable object that contains selected data items from the data source, for which
            the transformation function was applied.
        """
        return DataProvider(self._data_source, self._transform_func, indices)

    def get_length(self) -> Optional[int]:
        """
        Tries to fetch length of the underlying dataset.
        :return: The length of the data_source if __len__() is implemented for it, and None otherwise.
        """
        if hasattr(self._data_source, "__len__"):
            return self._data_source.__len__()
        return None

    def get_batch_size(self) -> Optional[int]:
        """
        Tries to fetch batch size of the underlying dataset.
        :return: The value of batch_size or _batch_size attributes of the data_source if exist, and None otherwise.
        """
        if hasattr(self._data_source, "batch_size"):  # Torch dataloader
            return self._data_source.batch_size
        if hasattr(self._data_source, "_batch_size"):  # TF dataloader
            return self._data_source._batch_size
        return None

class Dataset(Generic[DataItem, ModelInput]):
    """
    Wrapper for passing custom user datasets into NNCF algorithms.

    This class defines the interface by which compression algorithms
    retrieve data items from the passed data source object. These data items are used
    for different purposes, for example, model inference and model validation, based
    on the choice of the exact compression algorithm.

    If the data item has been returned from the data source per iteration and it cannot be
    used as input for model inference, the transformation function is used to extract the
    model's input from this data item. For example, in supervised learning, the data item
    usually contains both examples and labels. So transformation function should extract
    the examples from the data item.

    :param data_source: The iterable object serving as the source of data items.
    :param transform_func: The function that is used to extract the model's input
        from the data item. The data item here is the data item that is returned from
        the data source per iteration. This function should be passed when
        the data item cannot be directly used as model's input. If this is not specified, then the data item
        will be passed into the model as-is.
    """

    def __init__(
        self, data_source: Iterable[DataItem], transform_func: Optional[Callable[[DataItem], ModelInput]] = None
    ):
        self._data_source = data_source
        self._transform_func = transform_func

    def get_data(self, indices: Optional[List[int]] = None) -> Iterable[DataItem]:
        """
        Returns the iterable object that contains selected data items from the data source as-is.

        :param indices: The zero-based indices of data items that should be selected from
            the data source. The indices should be sorted in ascending order. If indices are
            not passed all data items are selected from the data source.
        :return: The iterable object that contains selected data items from the data source as-is.
        """
        return DataProvider(self._data_source, None, indices)

    def get_inference_data(self, indices: Optional[List[int]] = None) -> Iterable[ModelInput]:
        """
        Returns the iterable object that contains selected data items from the data source, for which
        the transformation function was applied. The item, which was returned per iteration from this
        iterable, can be used as the model's input for model inference.

        :param indices: The zero-based indices of data items that should be selected from
            the data source. The indices should be sorted in ascending order. If indices are
            not passed all data items are selected from the data source.
        :return: The iterable object that contains selected data items from the data source, for which
            the transformation function was applied.
        """
        return DataProvider(self._data_source, self._transform_func, indices)

    def get_length(self) -> Optional[int]:
        """
        Tries to fetch length of the underlying dataset.
        :return: The length of the data_source if __len__() is implemented for it, and None otherwise.
        """
        if hasattr(self._data_source, "__len__"):
            return self._data_source.__len__()
        return None

    def get_batch_size(self) -> Optional[int]:
        """
        Tries to fetch batch size of the underlying dataset.
        :return: The value of batch_size or _batch_size attributes of the data_source if exist, and None otherwise.
        """
        if hasattr(self._data_source, "batch_size"):  # Torch dataloader
            return self._data_source.batch_size
        if hasattr(self._data_source, "_batch_size"):  # TF dataloader
            return self._data_source._batch_size
        return None

class Dataset(Generic[DataItem, ModelInput]):
    """
    Wrapper for passing custom user datasets into NNCF algorithms.

    This class defines the interface by which compression algorithms
    retrieve data items from the passed data source object. These data items are used
    for different purposes, for example, model inference and model validation, based
    on the choice of the exact compression algorithm.

    If the data item has been returned from the data source per iteration and it cannot be
    used as input for model inference, the transformation function is used to extract the
    model's input from this data item. For example, in supervised learning, the data item
    usually contains both examples and labels. So transformation function should extract
    the examples from the data item.

    :param data_source: The iterable object serving as the source of data items.
    :param transform_func: The function that is used to extract the model's input
        from the data item. The data item here is the data item that is returned from
        the data source per iteration. This function should be passed when
        the data item cannot be directly used as model's input. If this is not specified, then the data item
        will be passed into the model as-is.
    """

    def __init__(
        self, data_source: Iterable[DataItem], transform_func: Optional[Callable[[DataItem], ModelInput]] = None
    ):
        self._data_source = data_source
        self._transform_func = transform_func

    def get_data(self, indices: Optional[List[int]] = None) -> Iterable[DataItem]:
        """
        Returns the iterable object that contains selected data items from the data source as-is.

        :param indices: The zero-based indices of data items that should be selected from
            the data source. The indices should be sorted in ascending order. If indices are
            not passed all data items are selected from the data source.
        :return: The iterable object that contains selected data items from the data source as-is.
        """
        return DataProvider(self._data_source, None, indices)

    def get_inference_data(self, indices: Optional[List[int]] = None) -> Iterable[ModelInput]:
        """
        Returns the iterable object that contains selected data items from the data source, for which
        the transformation function was applied. The item, which was returned per iteration from this
        iterable, can be used as the model's input for model inference.

        :param indices: The zero-based indices of data items that should be selected from
            the data source. The indices should be sorted in ascending order. If indices are
            not passed all data items are selected from the data source.
        :return: The iterable object that contains selected data items from the data source, for which
            the transformation function was applied.
        """
        return DataProvider(self._data_source, self._transform_func, indices)

    def get_length(self) -> Optional[int]:
        """
        Tries to fetch length of the underlying dataset.
        :return: The length of the data_source if __len__() is implemented for it, and None otherwise.
        """
        if hasattr(self._data_source, "__len__"):
            return self._data_source.__len__()
        return None

    def get_batch_size(self) -> Optional[int]:
        """
        Tries to fetch batch size of the underlying dataset.
        :return: The value of batch_size or _batch_size attributes of the data_source if exist, and None otherwise.
        """
        if hasattr(self._data_source, "batch_size"):  # Torch dataloader
            return self._data_source.batch_size
        if hasattr(self._data_source, "_batch_size"):  # TF dataloader
            return self._data_source._batch_size
        return None

