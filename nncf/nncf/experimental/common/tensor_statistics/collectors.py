class MeanNoOutliersAggregator(NoOutliersAggregatorBase):
    def _masked_aggregation_fn(
        self, stacked_samples: Tensor, mask: Tensor, axis: AggregationAxes, keepdims: bool
    ) -> Tensor:
        return fns.masked_mean(stacked_samples, axis=axis, mask=mask, keepdims=keepdims)

class MedianNoOutliersAggregator(NoOutliersAggregatorBase):
    def _masked_aggregation_fn(
        self, stacked_samples: Tensor, mask: Tensor, axis: AggregationAxes, keepdims: bool
    ) -> Tensor:
        return fns.masked_median(stacked_samples, axis=axis, mask=mask, keepdims=keepdims)

class TensorCollector:
    """
    Calculates statistics at given tensors according to registered statistic branches.
    Statistic branch consists of one reducer and one aggregator instance. TensorCollector
    applies a reducer on a correspondent inputs and then passes the one of the reduced tensors
    chosen by output port id to a correspondent aggregator for each registered statistic branch.
    Receives tensors by `register_input` method. Aggregated values as a TensorStatistic instance or
    a dict could be collected by `get_statistics` call.
    """

    def __init__(self, statistic_container: Optional[TensorStatistic] = None) -> None:
        self._reducers: Set[TensorReducerBase] = set()
        self._aggregators: Dict[Tuple[int, int, int], AggregatorBase] = {}
        self._stat_container_kwargs_map: Dict[str, Tuple[int, int, int]] = {}
        self._stat_container = statistic_container
        self._enabled = True

    @property
    def num_samples(self) -> Optional[int]:
        output = None
        for aggregator in self._aggregators.values():
            if aggregator.num_samples and output:
                output = max(output, aggregator.num_samples)
            else:
                output = aggregator.num_samples
        return output

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def reducers(self):
        return self._reducers.copy()

    @property
    def aggregators(self):
        return self._aggregators.copy()

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def register_statistic_branch(
        self,
        container_key: str,
        reducer: TensorReducerBase,
        aggregator: AggregatorBase,
        reducer_output_port_id: int = 0,
    ) -> None:
        """
        Registers statistic collection branch for a container key. Correspondent input will be reduced
        by given reducer and reduced value will be registered and aggregated by given aggregator.
        Passed container key should be unique for the TensorCollector instance.
        Passed aggregator instance should never be used twice for one TensorCollector instance.

        :param container_key: Container key to pass aggregated statistic to.
        :param reducer: TensorReducer instance for the statistic collection branch.
        :param aggregator: TensorAggregator instance for the statistic collection branch.
        :reducer_output_port_id: Reducer target output port id.
        """
        if container_key in self._stat_container_kwargs_map:
            raise nncf.InternalError(
                f"Two different statistic branches for one container key {container_key} are encountered"
            )
        if any(aggr is aggregator for aggr in self._aggregators.values()):
            raise nncf.InternalError(f"One aggregator instance {aggregator} for different branches is encountered")

        self._reducers.add(reducer)
        key = (hash(reducer), reducer_output_port_id, hash(aggregator))

        if key not in self._aggregators:
            self._aggregators[key] = aggregator
        self._stat_container_kwargs_map[container_key] = key

    def get_output_info(self, target_node_name: str, port_id: int) -> List[Tuple[int, List[str]]]:
        """
        Returns list of pairs of reducers names and correspondent output names.

        :param target_node_name: Target node name to assemble output name.
        :param port_id: Target node specific port id to assemble output name.
        :returns: List of pairs of reducers hashes and correspondent output names.
        """
        retval = []
        for reducer in self._reducers:
            retval.append((hash(reducer), reducer.get_output_names(target_node_name, port_id)))
        return retval

    def register_inputs(self, inputs: Dict[int, List[Tensor]]) -> None:
        """
        Registers given input in TensorCollector.

        :param inputs: Tensor inputs in format of dict where keys
            are reducer names and values are correspondent input tensors
        """
        if not self._enabled:
            return

        reduced_inputs = {}
        for reducer in self._reducers:
            reducer_hash = hash(reducer)
            input_ = inputs[reducer_hash]
            reduced_input = reducer(input_)
            if reduced_input is not None:
                reduced_inputs[reducer_hash] = reduced_input

        for (
            (reducer_hash, reducer_port_id, _),
            aggregator,
        ) in self._aggregators.items():
            if reducer_hash in reduced_inputs:
                aggregator.register_reduced_input(reduced_inputs[reducer_hash][reducer_port_id])

    def register_input_for_all_reducers(self, input_: Tensor) -> None:
        """
        Registers given input_ in each available statistic collection branch.

        :param input_: Tensor input to register.
        """
        self.register_inputs({hash(reducer): [input_] for reducer in self._reducers})

    def _aggregate(self) -> None:
        result = {}
        for (
            key,
            aggregator,
        ) in self._aggregators.items():
            val = aggregator.aggregate()
            result[key] = val
        return result

    def get_statistics(self) -> Union[TensorStatistic, Dict[str, Any]]:
        """
        Returns aggregated values in format of a TensorStatistic instance or
        a dict.

        :returns: Aggregated values.
        """

        aggregated_values = self._aggregate()
        kwargs = {}
        for container_key, branch_key in self._stat_container_kwargs_map.items():
            kwargs[container_key] = aggregated_values[branch_key]

        if not self._stat_container:
            return kwargs
        return self._build_statistic_container(self._stat_container, kwargs)

    def get_inplace_fn_info(self) -> List[Tuple[Any, int]]:
        """
        Returns necessary information to insert inplace operation into graph.

        :returns: necessary information to insert inplace operation into graph
            in format of pair of reducer builder and correspondent reducer output port id.
        """
        retval = []
        for reducer in self._reducers:
            if reducer.inplace:
                retval.append((reducer.get_inplace_fn(), reducer.output_port_id))
        return retval

    def any_stat_out_of_place(self) -> bool:
        """
        Returns True if any reducer is calculated out of place.

        :returns: True if any reducer is calculated out of place.
        """
        return any(not reducer.inplace for reducer in self._reducers)

    def replace_aggregator(self, key: Tuple[int, int, int], aggregator: AggregatorBase) -> None:
        """
        Friend method that replaces aggregator instance on equivalent one.
        Key should be valid for for given aggregator and a statistic branch
        with key should be present in TensorCollector.

        :param key: Statistic branch key.
        :param aggregator: Aggregator instance to replace existing instance by given key.
        """
        assert key in self._aggregators
        assert key[2] == hash(aggregator)
        self._aggregators[key] = aggregator

    def reset(self):
        for aggregator in self._aggregators.values():
            aggregator.reset()

    @staticmethod
    def get_tensor_collector_inputs(
        outputs: Dict[str, Tensor], output_info: List[Tuple[int, List[str]]]
    ) -> Dict[int, List[Tensor]]:
        """
        Static method that converts all model outputs and collected output_info
        to a layout required for `register_inputs` method. This method is not a part of
        `register_inputs` to avoid all inputs passing to `TensorCollector.register_inputs` method.

        :param outputs: Target model outputs.
        :param output_info: Output info collected by a `TensorCollector.get_output_info` method.
        :returns: Model outputs in a format required by `TensorCollector.register_inputs` method.
        """
        target_inputs = {}
        for reducer, names in output_info:
            target_inputs[reducer] = [outputs[name] for name in names]
        return target_inputs

    @staticmethod
    def _build_statistic_container(statistic_container_cls: Type[TensorStatistic], kwargs: Dict[Any, Any]):
        if issubclass(statistic_container_cls, MinMaxTensorStatistic):
            return statistic_container_cls(
                min_values=kwargs[MinMaxTensorStatistic.MIN_STAT], max_values=kwargs[MinMaxTensorStatistic.MAX_STAT]
            )
        if issubclass(statistic_container_cls, MeanTensorStatistic):
            return statistic_container_cls(
                mean_values=kwargs[MeanTensorStatistic.MEAN_STAT], shape=kwargs[MeanTensorStatistic.SHAPE_STAT]
            )
        if issubclass(statistic_container_cls, RawTensorStatistic):
            return statistic_container_cls(values=kwargs[RawTensorStatistic.VALUES_STATS])
        if issubclass(statistic_container_cls, MedianMADTensorStatistic):
            return statistic_container_cls(
                median_values=kwargs[MedianMADTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY][
                    MedianMADTensorStatistic.MEDIAN_VALUES_STAT
                ],
                mad_values=kwargs[MedianMADTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY][
                    MedianMADTensorStatistic.MAD_VALUES_STAT
                ],
            )
        if issubclass(statistic_container_cls, PercentileTensorStatistic):
            if PercentileTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY in kwargs:
                percentile_vs_values_dict = kwargs[PercentileTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY]
            else:
                percentile_vs_values_dict = {}
                for (_, percentile), value in kwargs.items():
                    percentile_vs_values_dict[percentile] = value
            return statistic_container_cls(percentile_vs_values_dict=percentile_vs_values_dict)
        raise nncf.InternalError(
            f"Statistic collector class {statistic_container_cls} is not supported by the TensorCollector class."
        )

