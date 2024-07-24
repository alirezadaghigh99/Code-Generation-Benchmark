class GroupedPooledEmbeddingsLookup(
    BaseEmbeddingLookup[KeyedJaggedTensor, torch.Tensor]
):
    """
    Lookup modules for Pooled embeddings (i.e EmbeddingBags)
    """

    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        device: Optional[torch.device] = None,
        pg: Optional[dist.ProcessGroup] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
        scale_weight_gradients: bool = True,
        sharding_type: Optional[ShardingType] = None,
    ) -> None:
        # TODO rename to _create_embedding_kernel
        def _create_lookup(
            config: GroupedEmbeddingConfig,
            device: Optional[torch.device] = None,
            sharding_type: Optional[ShardingType] = None,
        ) -> BaseEmbedding:
            if config.compute_kernel == EmbeddingComputeKernel.DENSE:
                return BatchedDenseEmbeddingBag(
                    config=config,
                    pg=pg,
                    device=device,
                    sharding_type=sharding_type,
                )
            elif config.compute_kernel == EmbeddingComputeKernel.FUSED:
                return BatchedFusedEmbeddingBag(
                    config=config,
                    pg=pg,
                    device=device,
                    sharding_type=sharding_type,
                )
            elif config.compute_kernel in {
                EmbeddingComputeKernel.KEY_VALUE,
            }:
                return KeyValueEmbeddingBag(
                    config=config,
                    pg=pg,
                    device=device,
                    sharding_type=sharding_type,
                )
            else:
                raise ValueError(
                    f"Compute kernel not supported {config.compute_kernel}"
                )

        super().__init__()
        self._emb_modules: nn.ModuleList = nn.ModuleList()
        for config in grouped_configs:
            self._emb_modules.append(_create_lookup(config, device, sharding_type))

        self._feature_splits: List[int] = []
        for config in grouped_configs:
            self._feature_splits.append(config.num_features())

        # return a dummy empty tensor when grouped_configs is empty
        self.register_buffer(
            "_dummy_embs_tensor",
            torch.empty(
                [0],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            ),
        )

        self.grouped_configs = grouped_configs
        self._feature_processor = feature_processor

        self._scale_gradient_factor: int = (
            dist.get_world_size(pg)
            if scale_weight_gradients and get_gradient_division()
            else 1
        )

    def prefetch(
        self,
        sparse_features: KeyedJaggedTensor,
        forward_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        def _need_prefetch(config: GroupedEmbeddingConfig) -> bool:
            for table in config.embedding_tables:
                if table.compute_kernel == EmbeddingComputeKernel.FUSED_UVM_CACHING:
                    return True
            return False

        if len(self._emb_modules) > 0:
            assert sparse_features is not None
            features_by_group = sparse_features.split(
                self._feature_splits,
            )
            for emb_op, features in zip(self._emb_modules, features_by_group):
                if not _need_prefetch(emb_op.config):
                    continue
                if (
                    isinstance(emb_op.emb_module, SplitTableBatchedEmbeddingBagsCodegen)
                    and not emb_op.emb_module.prefetch_pipeline
                ):
                    logging.error(
                        "Invalid setting on SplitTableBatchedEmbeddingBagsCodegen modules. prefetch_pipeline must be set to True.\n"
                        "If you don't turn on prefetch_pipeline, cache locations might be wrong in backward and can cause wrong results.\n"
                    )
                if hasattr(emb_op.emb_module, "prefetch"):
                    emb_op.emb_module.prefetch(
                        indices=features.values(),
                        offsets=features.offsets(),
                        forward_stream=forward_stream,
                        batch_size_per_feature_per_rank=(
                            features.stride_per_key_per_rank()
                            if features.variable_stride_per_key()
                            else None
                        ),
                    )

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        if len(self._emb_modules) > 0:
            assert sparse_features is not None
            features_by_group = sparse_features.split(
                self._feature_splits,
            )
            for config, emb_op, features in zip(
                self.grouped_configs, self._emb_modules, features_by_group
            ):
                if (
                    config.has_feature_processor
                    and self._feature_processor is not None
                    and isinstance(self._feature_processor, BaseGroupedFeatureProcessor)
                ):
                    features = self._feature_processor(features)

                if config.is_weighted:
                    features._weights = CommOpGradientScaling.apply(
                        features._weights, self._scale_gradient_factor
                    )

                embeddings.append(emb_op(features))

        dummy_embedding = (
            self._dummy_embs_tensor
            if sparse_features.variable_stride_per_key()
            else fx_wrap_tensor_view2d(
                self._dummy_embs_tensor, sparse_features.stride(), 0
            )
        )
        return embeddings_cat_empty_rank_handle(
            embeddings,
            dummy_embedding,
            dim=1,
        )

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()

        for emb_module in self._emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)

        return destination

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, Union[ShardedTensor, torch.Tensor]]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        m, u = _load_state_dict(self._emb_modules, state_dict)
        return _IncompatibleKeys(missing_keys=m, unexpected_keys=u)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_parameters for"
            "GroupedPooledEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_buffers for"
            "GroupedPooledEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)

    def named_parameters_by_table(
        self,
    ) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
        """
        Like named_parameters(), but yields table_name and embedding_weights which are wrapped in TableBatchedEmbeddingSlice.
        For a single table with multiple shards (i.e CW) these are combined into one table/weight.
        Used in composability.
        """
        for embedding_kernel in self._emb_modules:
            for (
                table_name,
                tbe_slice,
            ) in embedding_kernel.named_parameters_by_table():
                yield (table_name, tbe_slice)

    def flush(self) -> None:
        for emb_module in self._emb_modules:
            emb_module.flush()

    def purge(self) -> None:
        for emb_module in self._emb_modules:
            emb_module.purge()

class GroupedEmbeddingsLookup(BaseEmbeddingLookup[KeyedJaggedTensor, torch.Tensor]):
    """
    Lookup modules for Sequence embeddings (i.e Embeddings)
    """

    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        # TODO rename to _create_embedding_kernel
        def _create_lookup(
            config: GroupedEmbeddingConfig,
        ) -> BaseEmbedding:
            for table in config.embedding_tables:
                if table.compute_kernel == EmbeddingComputeKernel.FUSED_UVM_CACHING:
                    self._need_prefetch = True
            if config.compute_kernel == EmbeddingComputeKernel.DENSE:
                return BatchedDenseEmbedding(
                    config=config,
                    pg=pg,
                    device=device,
                )
            elif config.compute_kernel == EmbeddingComputeKernel.FUSED:
                return BatchedFusedEmbedding(
                    config=config,
                    pg=pg,
                    device=device,
                )
            elif config.compute_kernel in {
                EmbeddingComputeKernel.KEY_VALUE,
            }:
                return KeyValueEmbedding(
                    config=config,
                    pg=pg,
                    device=device,
                )
            else:
                raise ValueError(
                    f"Compute kernel not supported {config.compute_kernel}"
                )

        super().__init__()
        self._emb_modules: nn.ModuleList = nn.ModuleList()
        self._need_prefetch: bool = False
        for config in grouped_configs:
            self._emb_modules.append(_create_lookup(config))

        self._feature_splits: List[int] = []
        for config in grouped_configs:
            self._feature_splits.append(config.num_features())

        # return a dummy empty tensor when grouped_configs is empty
        self.register_buffer(
            "_dummy_embs_tensor",
            torch.empty(
                [0],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            ),
        )

        self.grouped_configs = grouped_configs

    def prefetch(
        self,
        sparse_features: KeyedJaggedTensor,
        forward_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if not self._need_prefetch:
            return
        if len(self._emb_modules) > 0:
            assert sparse_features is not None
            features_by_group = sparse_features.split(
                self._feature_splits,
            )
            for emb_op, features in zip(self._emb_modules, features_by_group):
                if (
                    isinstance(emb_op.emb_module, SplitTableBatchedEmbeddingBagsCodegen)
                    and not emb_op.emb_module.prefetch_pipeline
                ):
                    logging.error(
                        "Invalid setting on SplitTableBatchedEmbeddingBagsCodegen modules. prefetch_pipeline must be set to True.\n"
                        "If you don’t turn on prefetch_pipeline, cache locations might be wrong in backward and can cause wrong results.\n"
                    )
                if hasattr(emb_op.emb_module, "prefetch"):
                    emb_op.emb_module.prefetch(
                        indices=features.values(),
                        offsets=features.offsets(),
                        forward_stream=forward_stream,
                    )

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        features_by_group = sparse_features.split(
            self._feature_splits,
        )
        for emb_op, features in zip(self._emb_modules, features_by_group):
            embeddings.append(emb_op(features).view(-1))

        return embeddings_cat_empty_rank_handle(embeddings, self._dummy_embs_tensor)

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()

        for emb_module in self._emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)

        return destination

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, Union[torch.Tensor, ShardedTensor]]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        m, u = _load_state_dict(self._emb_modules, state_dict)
        return _IncompatibleKeys(missing_keys=m, unexpected_keys=u)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_parameters for"
            "GroupedEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_buffers for"
            "GroupedEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)

    def named_parameters_by_table(
        self,
    ) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
        """
        Like named_parameters(), but yields table_name and embedding_weights which are wrapped in TableBatchedEmbeddingSlice.
        For a single table with multiple shards (i.e CW) these are combined into one table/weight.
        Used in composability.
        """
        for embedding_kernel in self._emb_modules:
            for (
                table_name,
                tbe_slice,
            ) in embedding_kernel.named_parameters_by_table():
                yield (table_name, tbe_slice)

    def flush(self) -> None:
        for emb_module in self._emb_modules:
            emb_module.flush()

    def purge(self) -> None:
        for emb_module in self._emb_modules:
            emb_module.purge()

