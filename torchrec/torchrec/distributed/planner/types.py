class Storage:
    """
    Representation of the storage capacities of a hardware used in training.
    """

    hbm: int
    ddr: int

    def __add__(self, other: "Storage") -> "Storage":
        return Storage(
            hbm=self.hbm + other.hbm,
            ddr=self.ddr + other.ddr,
        )

    def __sub__(self, other: "Storage") -> "Storage":
        return Storage(
            hbm=self.hbm - other.hbm,
            ddr=self.ddr - other.ddr,
        )

    def __hash__(self) -> int:
        return hash((self.hbm, self.ddr))

    def fits_in(self, other: "Storage") -> bool:
        return self.hbm <= other.hbm and self.ddr <= other.ddr

class Perf:
    """
    Representation of the breakdown of the perf estimate a single shard of an
    embedding table.
    """

    fwd_compute: float
    fwd_comms: float
    bwd_compute: float
    bwd_comms: float
    prefetch_compute: float = 0.0

    @property
    def total(self) -> float:
        # When using embedding offload, there is a prefetch compute component. This
        # prefetch can overlap with fwd_compute + fwd_comm and dense fwd (some of it
        # overlaps with fwd_compute) and dense bwd. (fwd_compute and bwd_compute are
        # embedding fwd/bwd, nothing to do with dense). Only when prefetch is longer
        # than fwd_compute + dense_fwd + dense_bwd it will block bwd_compute. However,
        # we don't have an effective way to estimate dense fwd/bwd at this point, so our
        # cost model is too simplistic.  Instead prefetch is always considered blocking.
        #
        # Also note, measuring prefetch blocking can only be done after partitioning,
        # here are only have the per shard estimates.
        #
        # However adding a per-shard prefetch component to the cost model does have the
        # benefit that 1) it enables the ScaleupProposer to explore the trade off
        # between increasing cache sizes vs more difficult bin-packing constraints. 2)
        # it helps balance the prefetch compute across the ranks.
        return (
            self.fwd_compute
            + self.bwd_compute
            + self.fwd_comms
            + self.bwd_comms
            + self.prefetch_compute
        )

    def __add__(self, other: "Perf") -> "Perf":
        return Perf(
            fwd_compute=self.fwd_compute + other.fwd_compute,
            fwd_comms=self.fwd_comms + other.fwd_comms,
            bwd_compute=self.bwd_compute + other.bwd_compute,
            bwd_comms=self.bwd_comms + other.bwd_comms,
            prefetch_compute=self.prefetch_compute + other.prefetch_compute,
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.fwd_compute,
                self.fwd_comms,
                self.bwd_compute,
                self.bwd_comms,
                self.prefetch_compute,
            )
        )

class ShardingOption:
    """
    One way of sharding an embedding table. In the enumerator, we generate
    multiple sharding options per table, but in the planner output, there
    should only be one sharding option per table.

    Attributes:
        name (str): name of the sharding option.
        tensor (torch.Tensor): tensor of the sharding option. Usually on meta
            device.
        module (Tuple[str, nn.Module]): module and its fqn that contains the
            table.
        input_lengths (List[float]): list of pooling factors of the feature for
            the table.
        batch_size (int): batch size of training / eval job.
        sharding_type (str): sharding type of the table. Value of enum ShardingType.
        compute_kernel (str): compute kernel of the table. Value of enum
            EmbeddingComputeKernel.
        shards (List[Shard]): list of shards of the table.
        cache_params (Optional[CacheParams]): cache parameters to be used by this table.
            These are passed to FBGEMM's Split TBE kernel.
        enforce_hbm (Optional[bool]): whether to place all weights/momentums in HBM when
            using cache.
        stochastic_rounding (Optional[bool]): whether to do stochastic rounding. This is
            passed to FBGEMM's Split TBE kernel. Stochastic rounding is
            non-deterministic, but important to maintain accuracy in longer
            term with FP16 embedding tables.
        bounds_check_mode (Optional[BoundsCheckMode]): bounds check mode to be used by
            FBGEMM's Split TBE kernel. Bounds check means checking if values
            (i.e. row id) is within the table size. If row id exceeds table
            size, it will be set to 0.
        dependency (Optional[str]): dependency of the table. Related to
            Embedding tower.
        is_pooled (Optional[bool]): whether the table is pooled. Pooling can be
            sum pooling or mean pooling. Unpooled tables are also known as
            sequence embeddings.
        feature_names (Optional[List[str]]): list of feature names for this table.
        output_dtype (Optional[DataType]): output dtype to be used by this table.
            The default is FP32. If not None, the output dtype will also be used
            by the planner to produce a more balanced plan.
        key_value_params (Optional[KeyValueParams]): Params for SSD TBE, either
            for SSD or PS.
    """

    def __init__(
        self,
        name: str,
        tensor: torch.Tensor,
        module: Tuple[str, nn.Module],
        input_lengths: List[float],
        batch_size: int,
        sharding_type: str,
        partition_by: str,
        compute_kernel: str,
        shards: List[Shard],
        cache_params: Optional[CacheParams] = None,
        enforce_hbm: Optional[bool] = None,
        stochastic_rounding: Optional[bool] = None,
        bounds_check_mode: Optional[BoundsCheckMode] = None,
        dependency: Optional[str] = None,
        is_pooled: Optional[bool] = None,
        feature_names: Optional[List[str]] = None,
        output_dtype: Optional[DataType] = None,
        key_value_params: Optional[KeyValueParams] = None,
    ) -> None:
        self.name = name
        self._tensor = tensor
        self._module = module
        self.input_lengths = input_lengths
        self.batch_size = batch_size
        self.sharding_type = sharding_type
        self.partition_by = partition_by
        self.compute_kernel = compute_kernel
        # relevant to planner output, must be populated if sharding option
        # part of final solution
        self.shards = shards
        self.cache_params = cache_params
        self.enforce_hbm = enforce_hbm
        self.stochastic_rounding = stochastic_rounding
        self.bounds_check_mode = bounds_check_mode
        self.dependency = dependency
        self._is_pooled = is_pooled
        self.is_weighted: Optional[bool] = None
        self.feature_names: Optional[List[str]] = feature_names
        self.output_dtype: Optional[DataType] = output_dtype
        self.key_value_params: Optional[KeyValueParams] = key_value_params

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def module(self) -> Tuple[str, nn.Module]:
        return self._module

    @property
    def fqn(self) -> str:
        return self.module[0] + "." + self.name

    @property
    def cache_load_factor(self) -> Optional[float]:
        if self.cache_params is not None:
            return self.cache_params.load_factor
        return None

    @property
    def path(self) -> str:
        return self.module[0]

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    @property
    def num_inputs(self) -> int:
        return len(self.input_lengths)

    @property
    def total_storage(self) -> Storage:
        storage: Storage = Storage(hbm=0, ddr=0)
        for shard in self.shards:
            storage += cast(Storage, shard.storage)
        return storage

    @property
    def total_perf(self) -> float:
        perf: float = 0
        for shard in self.shards:
            # pyre-ignore: Undefined attribute [16]
            perf += shard.perf.total
        return perf

    @property
    def is_pooled(self) -> bool:
        if self._is_pooled is None:
            self._is_pooled = ShardingOption.module_pooled(self.module[1], self.name)
        return self._is_pooled

    @staticmethod
    def module_pooled(module: nn.Module, sharding_option_name: str) -> bool:
        """Determine if module pools output (e.g. EmbeddingBag) or uses unpooled/sequential output."""
        if isinstance(module, EmbeddingCollectionInterface) or isinstance(
            module, ManagedCollisionEmbeddingCollection
        ):
            return False

        for submodule in module.modules():
            if isinstance(submodule, EmbeddingCollectionInterface) or isinstance(
                submodule, ManagedCollisionEmbeddingCollection
            ):
                for name, _ in submodule.named_parameters():
                    if sharding_option_name in name:
                        return False

        return True

    def __hash__(self) -> int:
        return hash(
            (
                self.fqn,
                self.sharding_type,
                self.compute_kernel,
                tuple(self.shards),
                self.cache_params,
            )
        )

    def __deepcopy__(
        self, memo: Optional[Dict[int, "ShardingOption"]]
    ) -> "ShardingOption":
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k in ["_tensor", "_module"]:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    def __str__(self) -> str:
        str_obj: str = ""
        str_obj += f"name: {self.name}"
        str_obj += f"\nsharding type: {self.sharding_type}"
        str_obj += f"\ncompute kernel: {self.compute_kernel}"
        str_obj += f"\nnum shards: {len(self.shards)}"
        for shard in self.shards:
            str_obj += f"\n\t{str(shard)}"

        return str_obj

class ParameterConstraints:
    """
    Stores user provided constraints around the sharding plan.

    If provided, `pooling_factors`, `num_poolings`, and `batch_sizes` must match in
    length, as per sample.

    Attributes:
        sharding_types (Optional[List[str]]): sharding types allowed for the table.
            Values of enum ShardingType.
        compute_kernels (Optional[List[str]]): compute kernels allowed for the table.
            Values of enum EmbeddingComputeKernel.
        min_partition (Optional[int]): lower bound for dimension of column wise shards.
            Planner will search for the column wise shard dimension in the
            range of [min_partition, embedding_dim], as long as the column wise
            shard dimension divides embedding_dim and is divisible by 4. Used
            for column wise sharding only.
        pooling_factors (Optional[List[float]]): pooling factors for each feature of the
            table. This is the average number of values each sample has for
            the feature. Length of pooling_factors should match the number of
            features.
        num_poolings (OptionalList[float]]): number of poolings for each feature of the
            table. Length of num_poolings should match the number of features.
        batch_sizes (Optional[List[int]]): batch sizes for each feature of the table. Length
            of batch_sizes should match the number of features.
        is_weighted (Optional[bool]): whether the table is weighted.
        cache_params (Optional[CacheParams]): cache parameters to be used by this table.
            These are passed to FBGEMM's Split TBE kernel.
        enforce_hbm (Optional[bool]): whether to place all weights/momentums in HBM when
            using cache.
        stochastic_rounding (Optional[bool]): whether to do stochastic rounding. This is
            passed to FBGEMM's Split TBE kernel. Stochastic rounding is
            non-deterministic, but important to maintain accuracy in longer
            term with FP16 embedding tables.
        bounds_check_mode (Optional[BoundsCheckMode]): bounds check mode to be used by
            FBGEMM's Split TBE kernel. Bounds check means checking if values
            (i.e. row id) is within the table size. If row id exceeds table
            size, it will be set to 0.
        feature_names (Optional[List[str]]): list of feature names for this table.
        output_dtype (Optional[DataType]): output dtype to be used by this table.
            The default is FP32. If not None, the output dtype will also be used
            by the planner to produce a more balanced plan.
        device_group (Optional[str]): device group to be used by this table. It can be cpu
            or cuda. This specifies if the table should be placed on a cpu device
            or a gpu device.
        key_value_params (Optional[KeyValueParams]): key value params for SSD TBE, either for
            SSD or PS.
    """

    sharding_types: Optional[List[str]] = None
    compute_kernels: Optional[List[str]] = None
    min_partition: Optional[int] = None  # CW sharding, min CW dim to shard
    pooling_factors: List[float] = field(
        default_factory=lambda: [POOLING_FACTOR]
    )  # average number of embedding lookups required per sample
    num_poolings: Optional[List[float]] = None  # number of poolings per sample in batch
    batch_sizes: Optional[List[int]] = None  # batch size per input feature
    is_weighted: bool = False
    cache_params: Optional[CacheParams] = None
    enforce_hbm: Optional[bool] = None
    stochastic_rounding: Optional[bool] = None
    bounds_check_mode: Optional[BoundsCheckMode] = None
    feature_names: Optional[List[str]] = None
    output_dtype: Optional[DataType] = None
    device_group: Optional[str] = None
    key_value_params: Optional[KeyValueParams] = None

class DeviceHardware:
    """
    Representation of a device in a process group. 'perf' is an estimation of network,
    CPU, and storage usages.
    """

    rank: int
    storage: Storage
    perf: Perf

