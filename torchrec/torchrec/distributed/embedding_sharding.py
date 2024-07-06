def group_tables(
    tables_per_rank: List[List[ShardedEmbeddingTable]],
) -> List[List[GroupedEmbeddingConfig]]:
    """
    Groups tables by `DataType`, `PoolingType`, and `EmbeddingComputeKernel`.

    Args:
        tables_per_rank (List[List[ShardedEmbeddingTable]]): list of sharded embedding
            tables per rank with consistent weightedness.

    Returns:
        List[List[GroupedEmbeddingConfig]]: per rank list of GroupedEmbeddingConfig for features.
    """

    def _group_tables_per_rank(
        embedding_tables: List[ShardedEmbeddingTable],
    ) -> List[GroupedEmbeddingConfig]:
        grouped_embedding_configs: List[GroupedEmbeddingConfig] = []

        # We use different dim-bucketing policy for different cases.
        # If prefetch is off, all table (regardless of cache status or dimension) will be grouped together (SINGLE_BUCKET)
        # If prefetch is on,
        #     Cached vs noncached tables will be separated, even if they have the same dimension
        #     For two cached tables, if they have different dimension they shall be separated, otherwise they'll be grouped (ALL_BUCKETS)
        #     For two noncached tables, they'll be grouped regardless of dimension (SINGLE_BUCKET)
        prefetch_cached_dim_bucketer = EmbDimBucketer(
            list(filter(_prefetch_and_cached, embedding_tables)),
            EmbDimBucketerPolicy.ALL_BUCKETS,
        )
        non_prefetch_cached_dim_bucketer = EmbDimBucketer(
            list(filterfalse(_prefetch_and_cached, embedding_tables)),
            EmbDimBucketerPolicy.SINGLE_BUCKET,
        )

        # all embedding tables have the same weight status
        is_weighted = (
            embedding_tables[0].is_weighted if len(embedding_tables) > 0 else False
        )

        # Collect groups
        groups = defaultdict(list)
        grouping_keys = []
        for table in embedding_tables:
            bucketer = (
                prefetch_cached_dim_bucketer
                if _prefetch_and_cached(table)
                else non_prefetch_cached_dim_bucketer
            )
            group_fused_params = (
                _get_grouping_fused_params(table.fused_params, table.name) or {}
            )
            grouping_key = (
                table.data_type,
                table.pooling,
                table.has_feature_processor,
                tuple(sorted(group_fused_params.items())),
                _get_compute_kernel_type(table.compute_kernel),
                bucketer.get_bucket(table.local_cols, table.data_type),
                _prefetch_and_cached(table),
            )
            # micromanage the order of we traverse the groups to ensure backwards compatibility
            if grouping_key not in groups:
                grouping_keys.append(grouping_key)
            groups[grouping_key].append(table)

        for grouping_key in grouping_keys:
            (
                data_type,
                pooling,
                has_feature_processor,
                fused_params_tuple,
                compute_kernel_type,
                _,
                _,
            ) = grouping_key
            grouped_tables = groups[grouping_key]
            # remove non-native fused params
            per_tbe_fused_params = {
                k: v
                for k, v in fused_params_tuple
                if k not in ["_batch_key", USE_ONE_TBE_PER_TABLE]
            }
            cache_load_factor = _get_weighted_avg_cache_load_factor(grouped_tables)
            if cache_load_factor is not None:
                per_tbe_fused_params[CACHE_LOAD_FACTOR_STR] = cache_load_factor

            grouped_embedding_configs.append(
                GroupedEmbeddingConfig(
                    data_type=data_type,
                    pooling=pooling,
                    is_weighted=is_weighted,
                    has_feature_processor=has_feature_processor,
                    compute_kernel=compute_kernel_type,
                    embedding_tables=grouped_tables,
                    fused_params=per_tbe_fused_params,
                )
            )
        return grouped_embedding_configs

    table_weightedness = [
        table.is_weighted for tables in tables_per_rank for table in tables
    ]
    assert all(table_weightedness) or not any(table_weightedness)

    grouped_embedding_configs_by_rank: List[List[GroupedEmbeddingConfig]] = []
    for tables in tables_per_rank:
        grouped_embedding_configs = _group_tables_per_rank(tables)
        grouped_embedding_configs_by_rank.append(grouped_embedding_configs)

    return grouped_embedding_configs_by_rank

