class FeatureProcessedEmbeddingBagCollectionSharder(
    BaseEmbeddingSharder[FeatureProcessedEmbeddingBagCollection]
):
    def __init__(
        self,
        ebc_sharder: Optional[EmbeddingBagCollectionSharder] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._ebc_sharder: EmbeddingBagCollectionSharder = (
            ebc_sharder or EmbeddingBagCollectionSharder(self.qcomm_codecs_registry)
        )

    def shard(
        self,
        module: FeatureProcessedEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedFeatureProcessedEmbeddingBagCollection:

        if device is None:
            device = torch.device("cuda")

        return ShardedFeatureProcessedEmbeddingBagCollection(
            module,
            params,
            ebc_sharder=self._ebc_sharder,
            env=env,
            device=device,
        )

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        # TODO: to be deprecate after planner get cache_load_factor from ParameterConstraints
        return self._ebc_sharder.fused_params

    def shardable_parameters(
        self, module: FeatureProcessedEmbeddingBagCollection
    ) -> Dict[str, torch.nn.Parameter]:
        return self._ebc_sharder.shardable_parameters(module._embedding_bag_collection)

    @property
    def module_type(self) -> Type[FeatureProcessedEmbeddingBagCollection]:
        return FeatureProcessedEmbeddingBagCollection

    def sharding_types(self, compute_device_type: str) -> List[str]:
        if compute_device_type in {"mtia"}:
            return [ShardingType.TABLE_WISE.value, ShardingType.COLUMN_WISE.value]

        # No row wise because position weighted FP and RW don't play well together.
        types = [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.TABLE_COLUMN_WISE.value,
        ]

        return types

