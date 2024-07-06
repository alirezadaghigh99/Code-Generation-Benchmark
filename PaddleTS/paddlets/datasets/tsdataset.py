def load_from_dataframe(cls,
                            data: Union[pd.DataFrame, pd.Series],
                            time_col: Optional[str]=None,
                            value_cols: Optional[Union[List[str], str]]=None,
                            freq: Optional[Union[str, int]]=None,
                            drop_tail_nan: bool=False,
                            dtype: Optional[Union[type, Dict[str, type]]]=None
                            ) -> "TimeSeries":
        """
        Construct a TimeSeries object from the specified columns of a DataFrame

        Args:
            data(DataFrame|Series): A Pandas DataFrame or Series containing the time series data
            time_col(str|None): The name of time column, a Pandas DatetimeIndex or RangeIndex. 
                If not set, the DataFrame's index will be used.
            value_cols(list|str|None): The name of column or the list of columns from which to extract the time series data.
                If set to `None`, all columns except for the time column will be used as value columns.    
            freq(str|int|None): A string or int representing the Pandas DateTimeIndex's frequency or RangeIndex's step size
            drop_tail_nan(bool): Drop time series tail nan value or not, if True, drop all `Nan` value after the last `non-Nan` element in the current time series.
                eg: [nan, 3, 2, nan, nan] -> [nan, 3, 2], [3, 2, nan, nan] -> [3, 2], [nan, nan, nan] -> []
            dtype(np.dtype|type|dict): Use a numpy.dtype or Python type to cast entire TimeSeries object to the same type. 
                Alternatively, use {col: dtype, …}, where col is a column label and dtype is a numpy.dtype or 
                Python type to cast one or more of the DataFrame’s columns to column-specific types.

        Returns:
            TimeSeries object

        """
        #get data
        series_data = None
        if value_cols is None:
            if isinstance(data, pd.Series):
                series_data = data.copy()
            else:
                series_data = data.loc[:, data.columns != time_col].copy()
        else:
            series_data = data.loc[:, value_cols].copy()

        if isinstance(series_data, pd.DataFrame):
            raise_if_not(series_data.columns.is_unique,
                         "duplicated column names in the `data`!")
        #get time_col_vals
        if time_col:
            raise_if_not(
                time_col in data.columns,
                f"The time column: {time_col} doesn't exist in the `data`!")
            time_col_vals = data.loc[:, time_col]
        else:
            time_col_vals = data.index
        #Duplicated values or NaN are not allowed in the time column
        raise_if(time_col_vals.duplicated().any(),
                 "duplicated values in the time column!")
        #Try to convert to string and generate DatetimeIndex
        if np.issubdtype(time_col_vals.dtype, np.integer) and isinstance(freq,
                                                                         str):
            time_col_vals = time_col_vals.astype(str)
        #get time_index
        if np.issubdtype(time_col_vals.dtype, np.integer):
            if freq:
                #The type of freq should be int when the type of time_col is RangeIndex, which is set to 1 by default
                raise_if_not(
                    isinstance(freq, int) and freq >= 1,
                    "The type of freq should be int when the type of time_col is RangeIndex"
                )
            else:
                freq = 1
            start_idx, stop_idx = min(time_col_vals), max(time_col_vals) + freq
            # All integers in the range must be present
            raise_if((stop_idx - start_idx) / freq != len(data),
                     "The number of rows doesn't match with the RangeIndex!")
            time_index = pd.RangeIndex(
                start=start_idx, stop=stop_idx, step=freq)
        elif np.issubdtype(time_col_vals.dtype, np.object_) or \
            np.issubdtype(time_col_vals.dtype, np.datetime64):
            time_col_vals = pd.to_datetime(
                time_col_vals, infer_datetime_format=True)
            time_index = pd.DatetimeIndex(time_col_vals)
            if freq:
                #freq type needs to be string when time_col type is DatetimeIndex
                raise_if_not(
                    isinstance(freq, str),
                    "The type of `freq` should be `str` when the type of `time_col` is `DatetimeIndex`."
                )
            else:
                #If freq is not provided and automatic inference fail, throw exception
                freq = pd.infer_freq(time_index)
                raise_if(
                    freq is None,
                    "Failed to infer the `freq`. A valid `freq` is required.")
                if freq[0] == '-':
                    freq = freq[1:]
        else:
            raise_log(ValueError("The type of `time_col` is invalid."))
        if isinstance(series_data, pd.Series):
            series_data = series_data.to_frame()
        series_data.set_index(time_index, inplace=True)
        series_data.sort_index(inplace=True)
        ts = TimeSeries(series_data, freq)
        if drop_tail_nan:
            ts.drop_tail_nan()
        if dtype:
            ts.astype(dtype)
        return ts

def load_from_dataframe(cls,
                            data: Union[pd.DataFrame, pd.Series],
                            time_col: Optional[str]=None,
                            value_cols: Optional[Union[List[str], str]]=None,
                            freq: Optional[Union[str, int]]=None,
                            drop_tail_nan: bool=False,
                            dtype: Optional[Union[type, Dict[str, type]]]=None
                            ) -> "TimeSeries":
        """
        Construct a TimeSeries object from the specified columns of a DataFrame

        Args:
            data(DataFrame|Series): A Pandas DataFrame or Series containing the time series data
            time_col(str|None): The name of time column, a Pandas DatetimeIndex or RangeIndex. 
                If not set, the DataFrame's index will be used.
            value_cols(list|str|None): The name of column or the list of columns from which to extract the time series data.
                If set to `None`, all columns except for the time column will be used as value columns.    
            freq(str|int|None): A string or int representing the Pandas DateTimeIndex's frequency or RangeIndex's step size
            drop_tail_nan(bool): Drop time series tail nan value or not, if True, drop all `Nan` value after the last `non-Nan` element in the current time series.
                eg: [nan, 3, 2, nan, nan] -> [nan, 3, 2], [3, 2, nan, nan] -> [3, 2], [nan, nan, nan] -> []
            dtype(np.dtype|type|dict): Use a numpy.dtype or Python type to cast entire TimeSeries object to the same type. 
                Alternatively, use {col: dtype, …}, where col is a column label and dtype is a numpy.dtype or 
                Python type to cast one or more of the DataFrame’s columns to column-specific types.

        Returns:
            TimeSeries object

        """
        #get data
        series_data = None
        if value_cols is None:
            if isinstance(data, pd.Series):
                series_data = data.copy()
            else:
                series_data = data.loc[:, data.columns != time_col].copy()
        else:
            series_data = data.loc[:, value_cols].copy()

        if isinstance(series_data, pd.DataFrame):
            raise_if_not(series_data.columns.is_unique,
                         "duplicated column names in the `data`!")
        #get time_col_vals
        if time_col:
            raise_if_not(
                time_col in data.columns,
                f"The time column: {time_col} doesn't exist in the `data`!")
            time_col_vals = data.loc[:, time_col]
        else:
            time_col_vals = data.index
        #Duplicated values or NaN are not allowed in the time column
        raise_if(time_col_vals.duplicated().any(),
                 "duplicated values in the time column!")
        #Try to convert to string and generate DatetimeIndex
        if np.issubdtype(time_col_vals.dtype, np.integer) and isinstance(freq,
                                                                         str):
            time_col_vals = time_col_vals.astype(str)
        #get time_index
        if np.issubdtype(time_col_vals.dtype, np.integer):
            if freq:
                #The type of freq should be int when the type of time_col is RangeIndex, which is set to 1 by default
                raise_if_not(
                    isinstance(freq, int) and freq >= 1,
                    "The type of freq should be int when the type of time_col is RangeIndex"
                )
            else:
                freq = 1
            start_idx, stop_idx = min(time_col_vals), max(time_col_vals) + freq
            # All integers in the range must be present
            raise_if((stop_idx - start_idx) / freq != len(data),
                     "The number of rows doesn't match with the RangeIndex!")
            time_index = pd.RangeIndex(
                start=start_idx, stop=stop_idx, step=freq)
        elif np.issubdtype(time_col_vals.dtype, np.object_) or \
            np.issubdtype(time_col_vals.dtype, np.datetime64):
            time_col_vals = pd.to_datetime(
                time_col_vals, infer_datetime_format=True)
            time_index = pd.DatetimeIndex(time_col_vals)
            if freq:
                #freq type needs to be string when time_col type is DatetimeIndex
                raise_if_not(
                    isinstance(freq, str),
                    "The type of `freq` should be `str` when the type of `time_col` is `DatetimeIndex`."
                )
            else:
                #If freq is not provided and automatic inference fail, throw exception
                freq = pd.infer_freq(time_index)
                raise_if(
                    freq is None,
                    "Failed to infer the `freq`. A valid `freq` is required.")
                if freq[0] == '-':
                    freq = freq[1:]
        else:
            raise_log(ValueError("The type of `time_col` is invalid."))
        if isinstance(series_data, pd.Series):
            series_data = series_data.to_frame()
        series_data.set_index(time_index, inplace=True)
        series_data.sort_index(inplace=True)
        ts = TimeSeries(series_data, freq)
        if drop_tail_nan:
            ts.drop_tail_nan()
        if dtype:
            ts.astype(dtype)
        return ts

def load_from_dataframe(cls,
                            data: Union[pd.DataFrame, pd.Series],
                            time_col: Optional[str]=None,
                            value_cols: Optional[Union[List[str], str]]=None,
                            freq: Optional[Union[str, int]]=None,
                            drop_tail_nan: bool=False,
                            dtype: Optional[Union[type, Dict[str, type]]]=None
                            ) -> "TimeSeries":
        """
        Construct a TimeSeries object from the specified columns of a DataFrame

        Args:
            data(DataFrame|Series): A Pandas DataFrame or Series containing the time series data
            time_col(str|None): The name of time column, a Pandas DatetimeIndex or RangeIndex. 
                If not set, the DataFrame's index will be used.
            value_cols(list|str|None): The name of column or the list of columns from which to extract the time series data.
                If set to `None`, all columns except for the time column will be used as value columns.    
            freq(str|int|None): A string or int representing the Pandas DateTimeIndex's frequency or RangeIndex's step size
            drop_tail_nan(bool): Drop time series tail nan value or not, if True, drop all `Nan` value after the last `non-Nan` element in the current time series.
                eg: [nan, 3, 2, nan, nan] -> [nan, 3, 2], [3, 2, nan, nan] -> [3, 2], [nan, nan, nan] -> []
            dtype(np.dtype|type|dict): Use a numpy.dtype or Python type to cast entire TimeSeries object to the same type. 
                Alternatively, use {col: dtype, …}, where col is a column label and dtype is a numpy.dtype or 
                Python type to cast one or more of the DataFrame’s columns to column-specific types.

        Returns:
            TimeSeries object

        """
        #get data
        series_data = None
        if value_cols is None:
            if isinstance(data, pd.Series):
                series_data = data.copy()
            else:
                series_data = data.loc[:, data.columns != time_col].copy()
        else:
            series_data = data.loc[:, value_cols].copy()

        if isinstance(series_data, pd.DataFrame):
            raise_if_not(series_data.columns.is_unique,
                         "duplicated column names in the `data`!")
        #get time_col_vals
        if time_col:
            raise_if_not(
                time_col in data.columns,
                f"The time column: {time_col} doesn't exist in the `data`!")
            time_col_vals = data.loc[:, time_col]
        else:
            time_col_vals = data.index
        #Duplicated values or NaN are not allowed in the time column
        raise_if(time_col_vals.duplicated().any(),
                 "duplicated values in the time column!")
        #Try to convert to string and generate DatetimeIndex
        if np.issubdtype(time_col_vals.dtype, np.integer) and isinstance(freq,
                                                                         str):
            time_col_vals = time_col_vals.astype(str)
        #get time_index
        if np.issubdtype(time_col_vals.dtype, np.integer):
            if freq:
                #The type of freq should be int when the type of time_col is RangeIndex, which is set to 1 by default
                raise_if_not(
                    isinstance(freq, int) and freq >= 1,
                    "The type of freq should be int when the type of time_col is RangeIndex"
                )
            else:
                freq = 1
            start_idx, stop_idx = min(time_col_vals), max(time_col_vals) + freq
            # All integers in the range must be present
            raise_if((stop_idx - start_idx) / freq != len(data),
                     "The number of rows doesn't match with the RangeIndex!")
            time_index = pd.RangeIndex(
                start=start_idx, stop=stop_idx, step=freq)
        elif np.issubdtype(time_col_vals.dtype, np.object_) or \
            np.issubdtype(time_col_vals.dtype, np.datetime64):
            time_col_vals = pd.to_datetime(
                time_col_vals, infer_datetime_format=True)
            time_index = pd.DatetimeIndex(time_col_vals)
            if freq:
                #freq type needs to be string when time_col type is DatetimeIndex
                raise_if_not(
                    isinstance(freq, str),
                    "The type of `freq` should be `str` when the type of `time_col` is `DatetimeIndex`."
                )
            else:
                #If freq is not provided and automatic inference fail, throw exception
                freq = pd.infer_freq(time_index)
                raise_if(
                    freq is None,
                    "Failed to infer the `freq`. A valid `freq` is required.")
                if freq[0] == '-':
                    freq = freq[1:]
        else:
            raise_log(ValueError("The type of `time_col` is invalid."))
        if isinstance(series_data, pd.Series):
            series_data = series_data.to_frame()
        series_data.set_index(time_index, inplace=True)
        series_data.sort_index(inplace=True)
        ts = TimeSeries(series_data, freq)
        if drop_tail_nan:
            ts.drop_tail_nan()
        if dtype:
            ts.astype(dtype)
        return ts

