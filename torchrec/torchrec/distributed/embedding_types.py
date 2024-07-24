class ShardedEmbeddingTable(
    ShardedMetaConfig,
    EmbeddingAttributes,
    EmbeddingTableConfig,
):
    fused_params: Optional[Dict[str, Any]] = None

