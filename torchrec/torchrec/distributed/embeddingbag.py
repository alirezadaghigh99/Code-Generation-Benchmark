class EmbeddingBagCollectionSharder(BaseEmbeddingSharder[EmbeddingBagCollection]):
    """
    This implementation uses non-fused `EmbeddingBagCollection`
    """

    def shard(
        self,
        module: EmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedEmbeddingBagCollection:
        return ShardedEmbeddingBagCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            fused_params=self.fused_params,
            device=device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )

    def shardable_parameters(
        self, module: EmbeddingBagCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[0]: param
            for name, param in module.embedding_bags.named_parameters()
        }

    @property
    def module_type(self) -> Type[EmbeddingBagCollection]:
        return EmbeddingBagCollection

