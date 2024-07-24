class BenchmarkSet:
    """
    Defines a set of plots to produce.

    This will create plots for the cartesian product of the parameters in this class.
    See :class:`BenchmarkSuite` if you want a union of parameters.
    """

    name: str
    """ Name of this benchmark set. """

    datasets: Collection[DatasetFactory]
    """ Datasets to benchmark against. """

    models: Collection[ModelFactory]
    """ Models to benchmark. """

    plots: Collection[Plotter]
    """ Plots to generate. """

    do_compile: Collection[bool]
    """ Whether to use ``tf.function``. """

    do_optimise: Collection[bool]
    """ Whether to train the models. """

    do_predict: bool
    """ Whether to benchmark ``model.predict_f``. """

    do_posterior: bool
    """ Whether to benchmark ``model.posterior()``. """

    file_by: GroupingSpec
    """ How to split plots into different ``.png`` files. """

    column_by: GroupingSpec
    """ How to split plots into different columns within their file. """

    row_by: GroupingSpec
    """ How to split plots into different rows within their file. """

    line_by: Optional[GroupingSpec]
    """
    How to split data into different lines within a plot.

    If ``None`` data will be split by all columns not used by any of the other ``GroupingSpec``\s.
    """

    repetitions: int = 1
    """ Number of times to repeat benchmarks, to estimate noise. """

    def __post_init__(self) -> None:
        def has_unique_values(values: Collection[Any]) -> bool:
            return len(values) == len(set(values))

        def assert_unique_names(attr: str) -> None:
            names = [v.name for v in getattr(self, attr)]
            assert has_unique_values(names), f"'{attr}' must have unique names. Found: {names}."

        assert_unique_names("datasets")
        assert_unique_names("models")
        assert_unique_names("plots")

        def assert_unique_values(attr: str) -> None:
            values = getattr(self, attr)
            assert has_unique_values(values), f"'{attr}' must have unique values. Found: {values}."

        assert_unique_values("do_compile")
        assert_unique_values("do_optimise")

        def assert_disjoint_by(attr1: str, attr2: str) -> None:
            values1 = getattr(self, attr1).by
            values2 = getattr(self, attr2).by
            assert not (set(values1) & set(values2)), (
                f"'{attr1}.by' and '{attr2}.by' must be disjoint."
                f" Found: {attr1}.by={values1} and {attr2}.by={values2}."
            )

        assert_disjoint_by("file_by", "column_by")
        assert_disjoint_by("file_by", "row_by")
        assert_disjoint_by("column_by", "row_by")
        if self.line_by:
            assert_disjoint_by("file_by", "line_by")
            assert_disjoint_by("column_by", "line_by")
            assert_disjoint_by("row_by", "line_by")

        def assert_grouping_by(by: GroupingKey) -> None:
            assert (by in self.file_by.by) or (by in self.column_by.by) or (by in self.row_by.by), (
                f"Must group by '{by}' above the 'line' level. Found:"
                f" file_by={self.file_by.by},"
                f" column_by={self.column_by.by},"
                f" row_by={self.row_by.by}."
            )

        assert_grouping_by(GroupingKey.METRIC)
        assert_grouping_by(GroupingKey.PLOTTER)

    @property
    def safe_line_by(self) -> GroupingSpec:
        """
        Get ``line_by``, or a default value if ``line_by`` is ``None``.
        """
        if self.line_by is None:
            used_by: Set[GroupingKey] = (
                set(self.file_by.by) | set(self.column_by.by) | set(self.row_by.by)
            )
            line_by = set(GroupingKey) - used_by
            sorted_line_by: Sequence[GroupingKey] = sorted(line_by, key=lambda k: k.key_cost)  # type: ignore[arg-type] # for lambda
            return GroupingSpec(sorted_line_by, minimise=True)

        return self.line_by

    def get_tasks(self) -> Collection[BenchmarkTask]:
        """
        Compute ``BenchmarkTask`` objects for the cartesian product of the parameters of this
        object.
        """
        result: List[BenchmarkTask] = []
        for dataset in self.datasets:
            dataset_name = dataset.name
            for model in self.models:
                if not model.dataset_req.satisfied(dataset.tags):
                    continue

                model_name = model.name
                for do_compile in self.do_compile:
                    for do_optimise in self.do_optimise:
                        result.append(
                            BenchmarkTask(
                                dataset_name,
                                model_name,
                                do_compile,
                                do_optimise,
                                self.do_predict,
                                self.do_posterior,
                                self.repetitions,
                            )
                        )
        return result

    def filter_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter a dataframe for any metrics that are not relevant to this :class:`BenchmarkSet`.
        """
        dataset_names = set(d.name for d in self.datasets)
        model_names = set(m.name for m in self.models)
        return metrics_df[
            metrics_df.dataset.isin(dataset_names)
            & metrics_df.model.isin(model_names)
            & metrics_df.do_compile.isin(self.do_compile)
            & metrics_df.do_optimise.isin(self.do_optimise)
            & (metrics_df.repetition < self.repetitions)
        ]

