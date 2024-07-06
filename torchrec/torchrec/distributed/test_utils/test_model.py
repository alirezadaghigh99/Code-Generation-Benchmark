def generate(
        batch_size: int,
        world_size: int,
        num_float_features: int,
        tables: Union[
            List[EmbeddingTableConfig], List[EmbeddingBagConfig], List[EmbeddingConfig]
        ],
        weighted_tables: Union[
            List[EmbeddingTableConfig], List[EmbeddingBagConfig], List[EmbeddingConfig]
        ],
        pooling_avg: int = 10,
        dedup_tables: Optional[
            Union[
                List[EmbeddingTableConfig],
                List[EmbeddingBagConfig],
                List[EmbeddingConfig],
            ]
        ] = None,
        variable_batch_size: bool = False,
        long_indices: bool = True,
        tables_pooling: Optional[List[int]] = None,
        weighted_tables_pooling: Optional[List[int]] = None,
        randomize_indices: bool = True,
        device: Optional[torch.device] = None,
        max_feature_lengths: Optional[List[int]] = None,
    ) -> Tuple["ModelInput", List["ModelInput"]]:
        """
        Returns a global (single-rank training) batch
        and a list of local (multi-rank training) batches of world_size.
        """
        batch_size_by_rank = [batch_size] * world_size
        if variable_batch_size:
            batch_size_by_rank = [
                batch_size_by_rank[r] - r if batch_size_by_rank[r] - r > 0 else 1
                for r in range(world_size)
            ]

        def _validate_pooling_factor(
            tables: Union[
                List[EmbeddingTableConfig],
                List[EmbeddingBagConfig],
                List[EmbeddingConfig],
            ],
            pooling_factor: Optional[List[int]],
        ) -> None:
            if pooling_factor and len(pooling_factor) != len(tables):
                raise ValueError(
                    "tables_pooling and tables must have the same length. "
                    f"Got {len(pooling_factor)} and {len(tables)}."
                )

        _validate_pooling_factor(tables, tables_pooling)
        _validate_pooling_factor(weighted_tables, weighted_tables_pooling)

        idlist_features_to_num_embeddings = {}
        idlist_features_to_pooling_factor = {}
        idlist_features_to_max_length = {}
        feature_idx = 0
        for idx in range(len(tables)):
            for feature in tables[idx].feature_names:
                idlist_features_to_num_embeddings[feature] = tables[idx].num_embeddings
                idlist_features_to_max_length[feature] = (
                    max_feature_lengths[feature_idx] if max_feature_lengths else None
                )
                if tables_pooling is not None:
                    idlist_features_to_pooling_factor[feature] = tables_pooling[idx]
                feature_idx += 1

        idlist_features = list(idlist_features_to_num_embeddings.keys())
        idscore_features = [
            feature for table in weighted_tables for feature in table.feature_names
        ]

        idlist_ind_ranges = list(idlist_features_to_num_embeddings.values())
        idscore_ind_ranges = [table.num_embeddings for table in weighted_tables]

        idlist_pooling_factor = list(idlist_features_to_pooling_factor.values())
        idscore_pooling_factor = weighted_tables_pooling

        idlist_max_lengths = list(idlist_features_to_max_length.values())

        # Generate global batch.
        global_idlist_lengths = []
        global_idlist_indices = []
        global_idscore_lengths = []
        global_idscore_indices = []
        global_idscore_weights = []

        for idx in range(len(idlist_ind_ranges)):
            ind_range = idlist_ind_ranges[idx]
            if idlist_pooling_factor:
                lengths_ = torch.max(
                    torch.normal(
                        idlist_pooling_factor[idx],
                        idlist_pooling_factor[idx] / 10,
                        [batch_size * world_size],
                        device=device,
                    ),
                    torch.tensor(1.0, device=device),
                ).int()
            else:
                lengths_ = torch.abs(
                    torch.randn(batch_size * world_size, device=device) + pooling_avg,
                ).int()

            if idlist_max_lengths[idx]:
                lengths_ = torch.clamp(lengths_, max=idlist_max_lengths[idx])

            if variable_batch_size:
                lengths = torch.zeros(batch_size * world_size, device=device).int()
                for r in range(world_size):
                    lengths[r * batch_size : r * batch_size + batch_size_by_rank[r]] = (
                        lengths_[
                            r * batch_size : r * batch_size + batch_size_by_rank[r]
                        ]
                    )
            else:
                lengths = lengths_

            num_indices = cast(int, torch.sum(lengths).item())
            if randomize_indices:
                indices = torch.randint(
                    0,
                    ind_range,
                    (num_indices,),
                    dtype=torch.long if long_indices else torch.int32,
                    device=device,
                )
            else:
                indices = torch.zeros(
                    (num_indices),
                    dtype=torch.long if long_indices else torch.int32,
                    device=device,
                )
            global_idlist_lengths.append(lengths)
            global_idlist_indices.append(indices)
        global_idlist_kjt = KeyedJaggedTensor(
            keys=idlist_features,
            values=torch.cat(global_idlist_indices),
            lengths=torch.cat(global_idlist_lengths),
        )

        for idx in range(len(idscore_ind_ranges)):
            ind_range = idscore_ind_ranges[idx]
            lengths_ = torch.abs(
                torch.randn(batch_size * world_size, device=device)
                + (
                    idscore_pooling_factor[idx]
                    if idscore_pooling_factor
                    else pooling_avg
                )
            ).int()
            if variable_batch_size:
                lengths = torch.zeros(batch_size * world_size, device=device).int()
                for r in range(world_size):
                    lengths[r * batch_size : r * batch_size + batch_size_by_rank[r]] = (
                        lengths_[
                            r * batch_size : r * batch_size + batch_size_by_rank[r]
                        ]
                    )
            else:
                lengths = lengths_
            num_indices = cast(int, torch.sum(lengths).item())
            if randomize_indices:
                indices = torch.randint(
                    0,
                    ind_range,
                    (num_indices,),
                    dtype=torch.long if long_indices else torch.int32,
                    device=device,
                )
            else:
                indices = torch.zeros(
                    (num_indices),
                    dtype=torch.long if long_indices else torch.int32,
                    device=device,
                )
            weights = torch.rand((num_indices,), device=device)
            global_idscore_lengths.append(lengths)
            global_idscore_indices.append(indices)
            global_idscore_weights.append(weights)
        global_idscore_kjt = (
            KeyedJaggedTensor(
                keys=idscore_features,
                values=torch.cat(global_idscore_indices),
                lengths=torch.cat(global_idscore_lengths),
                weights=torch.cat(global_idscore_weights),
            )
            if global_idscore_indices
            else None
        )

        if randomize_indices:
            global_float = torch.rand(
                (batch_size * world_size, num_float_features), device=device
            )
            global_label = torch.rand(batch_size * world_size, device=device)
        else:
            global_float = torch.zeros(
                (batch_size * world_size, num_float_features), device=device
            )
            global_label = torch.zeros(batch_size * world_size, device=device)

        # Split global batch into local batches.
        local_inputs = []
        for r in range(world_size):
            local_idlist_lengths = []
            local_idlist_indices = []
            local_idscore_lengths = []
            local_idscore_indices = []
            local_idscore_weights = []

            for lengths, indices in zip(global_idlist_lengths, global_idlist_indices):
                local_idlist_lengths.append(
                    lengths[r * batch_size : r * batch_size + batch_size_by_rank[r]]
                )
                lengths_cumsum = [0] + lengths.view(world_size, -1).sum(dim=1).cumsum(
                    dim=0
                ).tolist()
                local_idlist_indices.append(
                    indices[lengths_cumsum[r] : lengths_cumsum[r + 1]]
                )

            for lengths, indices, weights in zip(
                global_idscore_lengths, global_idscore_indices, global_idscore_weights
            ):
                local_idscore_lengths.append(
                    lengths[r * batch_size : r * batch_size + batch_size_by_rank[r]]
                )
                lengths_cumsum = [0] + lengths.view(world_size, -1).sum(dim=1).cumsum(
                    dim=0
                ).tolist()
                local_idscore_indices.append(
                    indices[lengths_cumsum[r] : lengths_cumsum[r + 1]]
                )
                local_idscore_weights.append(
                    weights[lengths_cumsum[r] : lengths_cumsum[r + 1]]
                )

            local_idlist_kjt = KeyedJaggedTensor(
                keys=idlist_features,
                values=torch.cat(local_idlist_indices),
                lengths=torch.cat(local_idlist_lengths),
            )

            local_idscore_kjt = (
                KeyedJaggedTensor(
                    keys=idscore_features,
                    values=torch.cat(local_idscore_indices),
                    lengths=torch.cat(local_idscore_lengths),
                    weights=torch.cat(local_idscore_weights),
                )
                if local_idscore_indices
                else None
            )

            local_input = ModelInput(
                float_features=global_float[r * batch_size : (r + 1) * batch_size],
                idlist_features=local_idlist_kjt,
                idscore_features=local_idscore_kjt,
                label=global_label[r * batch_size : (r + 1) * batch_size],
            )
            local_inputs.append(local_input)

        return (
            ModelInput(
                float_features=global_float,
                idlist_features=global_idlist_kjt,
                idscore_features=global_idscore_kjt,
                label=global_label,
            ),
            local_inputs,
        )

