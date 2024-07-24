class TestEmbeddingCollectionSharder(EmbeddingCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        use_index_dedup: bool = False,
    ) -> None:
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

        qcomm_codecs_registry = {}
        if qcomms_config is not None:
            qcomm_codecs_registry = get_qcomm_codecs_registry(qcomms_config)

        if fused_params is None:
            fused_params = {}
        if "learning_rate" not in fused_params:
            fused_params["learning_rate"] = 0.1

        super().__init__(
            fused_params=fused_params,
            qcomm_codecs_registry=qcomm_codecs_registry,
            use_index_dedup=use_index_dedup,
        )

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]

class TestSequenceSparseNN(TestSparseNNBase):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        weighted_tables: Optional[List[EmbeddingConfig]] = None,
        num_float_features: int = 10,
        embedding_groups: Optional[Dict[str, List[str]]] = None,
        dense_device: Optional[torch.device] = None,
        sparse_device: Optional[torch.device] = None,
        feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
    ) -> None:
        super().__init__(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(Optional[List[BaseEmbeddingConfig]], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=dense_device,
            sparse_device=sparse_device,
        )
        if embedding_groups is None:
            embedding_groups = {}

        self.dense = TestDenseArch(
            device=dense_device, num_float_features=num_float_features
        )
        self.sparse = TestSequenceSparseArch(
            tables,
            list(embedding_groups.values())[0] if embedding_groups.values() else [],
            device=sparse_device,
        )
        self.over = TestSequenceOverArch(tables=tables, device=dense_device)

    def forward(
        self,
        input: ModelInput,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dense_r = self.dense(input.float_features)
        # multiply the sparse output by 10 since the model output is not sensitive to the
        # embedding output. It won't catch the unexpected embedding output without this
        sparse_r = 10 * self.sparse(input.idlist_features, input.float_features.size(0))
        over_r = self.over(dense_r, sparse_r)
        pred = torch.sigmoid(torch.mean(over_r, dim=1))
        if self.training:
            return (
                torch.nn.functional.binary_cross_entropy_with_logits(pred, input.label),
                pred,
            )
        else:
            return pred

