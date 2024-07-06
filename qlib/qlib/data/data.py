def features(
        self,
        instruments,
        fields,
        start_time=None,
        end_time=None,
        freq="day",
        disk_cache=None,
        inst_processors=[],
    ):
        """
        Parameters
        ----------
        disk_cache : int
            whether to skip(0)/use(1)/replace(2) disk_cache


        This function will try to use cache method which has a keyword `disk_cache`,
        and will use provider method if a type error is raised because the DatasetD instance
        is a provider class.
        """
        disk_cache = C.default_disk_cache if disk_cache is None else disk_cache
        fields = list(fields)  # In case of tuple.
        try:
            return DatasetD.dataset(
                instruments, fields, start_time, end_time, freq, disk_cache, inst_processors=inst_processors
            )
        except TypeError:
            return DatasetD.dataset(instruments, fields, start_time, end_time, freq, inst_processors=inst_processors)

def expression(self, instrument, field, start_time=None, end_time=None, freq="day") -> pd.Series:
        """Get Expression data.

        The responsibility of `expression`
        - parse the `field` and `load` the according data.
        - When loading the data, it should handle the time dependency of the data. `get_expression_instance` is commonly used in this method

        Parameters
        ----------
        instrument : str
            a certain instrument.
        field : str
            a certain field of feature.
        start_time : str
            start of the time range.
        end_time : str
            end of the time range.
        freq : str
            time frequency, available: year/quarter/month/week/day.

        Returns
        -------
        pd.Series
            data of a certain expression

            The data has two types of format

            1) expression with datetime index

            2) expression with integer index

                - because the datetime is not as good as
        """
        raise NotImplementedError("Subclass of ExpressionProvider must implement `Expression` method")

