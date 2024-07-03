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