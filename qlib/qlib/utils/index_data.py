class SingleData(IndexData):
    def __init__(
        self, data: Union[int, float, np.number, list, dict, pd.Series] = [], index: Union[List, pd.Index, Index] = []
    ):
        """A data structure of index and numpy data.
        It's used to replace pd.Series due to high-speed.

        Parameters
        ----------
        data : Union[int, float, np.number, list, dict, pd.Series]
            the input data
        index : Union[list, pd.Index]
            the index of data.
            empty list indicates that auto filling the index to the length of data
        """
        # for special data type
        if isinstance(data, dict):
            assert len(index) == 0
            if len(data) > 0:
                index, data = zip(*data.items())
            else:
                index, data = [], []
        elif isinstance(data, pd.Series):
            assert len(index) == 0
            index, data = data.index, data.values
        elif isinstance(data, (int, float, np.number)):
            data = [data]
        super().__init__(data, index)
        assert self.ndim == 1

    def _align_indices(self, other):
        if self.index == other.index:
            return other
        elif set(self.index) == set(other.index):
            return other.reindex(self.index)
        else:
            raise ValueError(
                f"The indexes of self and other do not meet the requirements of the four arithmetic operations"
            )

    def reindex(self, index: Index, fill_value=np.NaN) -> SingleData:
        """reindex data and fill the missing value with np.NaN.

        Parameters
        ----------
        new_index : list
            new index
        fill_value:
            what value to fill if index is missing

        Returns
        -------
        SingleData
            reindex data
        """
        # TODO: This method can be more general
        if self.index == index:
            return self
        tmp_data = np.full(len(index), fill_value, dtype=np.float64)
        for index_id, index_item in enumerate(index):
            try:
                tmp_data[index_id] = self.loc[index_item]
            except KeyError:
                pass
        return SingleData(tmp_data, index)

    def add(self, other: SingleData, fill_value=0):
        # TODO: add and __add__ are a little confusing.
        # This could be a more general
        common_index = self.index | other.index
        common_index, _ = common_index.sort()
        tmp_data1 = self.reindex(common_index, fill_value)
        tmp_data2 = other.reindex(common_index, fill_value)
        return tmp_data1.fillna(fill_value) + tmp_data2.fillna(fill_value)

    def to_dict(self):
        """convert SingleData to dict.

        Returns
        -------
        dict
            data with the dict format.
        """
        return dict(zip(self.index, self.data.tolist()))

    def to_series(self):
        return pd.Series(self.data, index=self.index)

    def __repr__(self) -> str:
        return str(pd.Series(self.data, index=self.index))

class MultiData(IndexData):
    def __init__(
        self,
        data: Union[int, float, np.number, list] = [],
        index: Union[List, pd.Index, Index] = [],
        columns: Union[List, pd.Index, Index] = [],
    ):
        """A data structure of index and numpy data.
        It's used to replace pd.DataFrame due to high-speed.

        Parameters
        ----------
        data : Union[list, np.ndarray]
            the dim of data must be 2.
        index : Union[List, pd.Index, Index]
            the index of data.
        columns: Union[List, pd.Index, Index]
            the columns of data.
        """
        if isinstance(data, pd.DataFrame):
            index, columns, data = data.index, data.columns, data.values
        super().__init__(data, index, columns)
        assert self.ndim == 2

    def _align_indices(self, other):
        if self.indices == other.indices:
            return other
        else:
            raise ValueError(
                f"The indexes of self and other do not meet the requirements of the four arithmetic operations"
            )

    def __repr__(self) -> str:
        return str(pd.DataFrame(self.data, index=self.index, columns=self.columns))

