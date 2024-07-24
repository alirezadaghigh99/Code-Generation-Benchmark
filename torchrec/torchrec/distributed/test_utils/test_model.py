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

class TestModelWithPreproc(nn.Module):
    """
    Basic module with up to 3 preproc modules:
    - preproc on idlist_features for non-weighted EBC
    - preproc on idscore_features for weighted EBC
    - optional preproc on model input shared by both EBCs

    Args:
        tables,
        weighted_tables,
        device,
        preproc_module,
        num_float_features,
        run_preproc_inline,

    Example:
        >>> TestModelWithPreproc(tables, weighted_tables, device)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        device: torch.device,
        preproc_module: Optional[nn.Module] = None,
        num_float_features: int = 10,
        run_preproc_inline: bool = False,
    ) -> None:
        super().__init__()
        self.dense = TestDenseArch(num_float_features, device)

        self.ebc: EmbeddingBagCollection = EmbeddingBagCollection(
            tables=tables,
            device=device,
        )
        self.weighted_ebc = EmbeddingBagCollection(
            tables=weighted_tables,
            is_weighted=True,
            device=device,
        )
        self.preproc_nonweighted = TestPreprocNonWeighted()
        self.preproc_weighted = TestPreprocWeighted()
        self._preproc_module = preproc_module
        self._run_preproc_inline = run_preproc_inline

    def forward(
        self,
        input: ModelInput,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs preprco for EBC and weighted EBC, optionally runs preproc for input

        Args:
            input
        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        modified_input = input

        if self._preproc_module is not None:
            modified_input = self._preproc_module(modified_input)
        elif self._run_preproc_inline:
            modified_input.idlist_features = KeyedJaggedTensor.from_lengths_sync(
                modified_input.idlist_features.keys(),
                modified_input.idlist_features.values(),
                modified_input.idlist_features.lengths(),
            )

        modified_idlist_features = self.preproc_nonweighted(
            modified_input.idlist_features
        )
        modified_idscore_features = self.preproc_weighted(
            modified_input.idscore_features
        )
        ebc_out = self.ebc(modified_idlist_features[0])
        weighted_ebc_out = self.weighted_ebc(modified_idscore_features[0])

        pred = torch.cat([ebc_out.values(), weighted_ebc_out.values()], dim=1)
        return pred.sum(), pred

class TestSparseNN(TestSparseNNBase, CopyableMixin):
    """
    Simple version of a SparseNN model.

    Args:
        tables: List[EmbeddingBagConfig],
        weighted_tables: Optional[List[EmbeddingBagConfig]],
        embedding_groups: Optional[Dict[str, List[str]]],
        dense_device: Optional[torch.device],
        sparse_device: Optional[torch.device],

    Call Args:
        input: ModelInput,

    Returns:
        torch.Tensor

    Example::

        TestSparseNN()
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        num_float_features: int = 10,
        weighted_tables: Optional[List[EmbeddingBagConfig]] = None,
        embedding_groups: Optional[Dict[str, List[str]]] = None,
        dense_device: Optional[torch.device] = None,
        sparse_device: Optional[torch.device] = None,
        max_feature_lengths_list: Optional[List[Dict[str, int]]] = None,
        feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
        over_arch_clazz: Type[nn.Module] = TestOverArch,
        preproc_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(Optional[List[BaseEmbeddingConfig]], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=dense_device,
            sparse_device=sparse_device,
        )
        if weighted_tables is None:
            weighted_tables = []
        self.dense = TestDenseArch(num_float_features, dense_device)
        self.sparse = TestSparseArch(
            tables,
            weighted_tables,
            sparse_device,
            max_feature_lengths_list if max_feature_lengths_list is not None else None,
        )

        embedding_names = (
            list(embedding_groups.values())[0] if embedding_groups else None
        )
        self._embedding_names: List[str] = (
            embedding_names
            if embedding_names
            else [feature for table in tables for feature in table.feature_names]
        )
        self._weighted_features: List[str] = [
            feature for table in weighted_tables for feature in table.feature_names
        ]
        self.over: nn.Module = over_arch_clazz(
            tables, weighted_tables, embedding_names, dense_device
        )
        self.register_buffer(
            "dummy_ones",
            torch.ones(1, device=dense_device),
        )
        self.preproc_module = preproc_module

    def sparse_forward(self, input: ModelInput) -> KeyedTensor:
        return self.sparse(
            features=input.idlist_features,
            weighted_features=input.idscore_features,
            batch_size=input.float_features.size(0),
        )

    def dense_forward(
        self, input: ModelInput, sparse_output: KeyedTensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dense_r = self.dense(input.float_features)
        over_r = self.over(dense_r, sparse_output)
        pred = torch.sigmoid(torch.mean(over_r, dim=1)) + self.dummy_ones
        if self.training:
            return (
                torch.nn.functional.binary_cross_entropy_with_logits(pred, input.label),
                pred,
            )
        else:
            return pred

    def forward(
        self,
        input: ModelInput,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.preproc_module:
            input = self.preproc_module(input)
        return self.dense_forward(input, self.sparse_forward(input))

class TestEBCSharder(EmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        if fused_params is None:
            fused_params = {}

        self._sharding_type = sharding_type
        self._kernel_type = kernel_type
        super().__init__(fused_params, qcomm_codecs_registry)

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

class TestNegSamplingModule(torch.nn.Module):
    """
    Basic module to simulate feature augmentation preproc (e.g. neg sampling) for testing

    Args:
        extra_input
        has_params

    Example:
        >>> preproc = TestNegSamplingModule(extra_input)
        >>> out = preproc(in)

    Returns:
        ModelInput
    """

    def __init__(
        self,
        extra_input: ModelInput,
        has_params: bool = False,
    ) -> None:
        super().__init__()
        self._extra_input = extra_input
        if has_params:
            self._linear: nn.Module = nn.Linear(30, 30)

    def forward(self, input: ModelInput) -> ModelInput:
        """
        Appends extra features to model input

        Args:
            input
        Returns:
            ModelInput
        """

        # merge extra input
        modified_input = copy.deepcopy(input)

        # dim=0 (batch dimensions) increases by self._extra_input.float_features.shape[0]
        modified_input.float_features = torch.concat(
            (modified_input.float_features, self._extra_input.float_features), dim=0
        )

        # stride will be same but features will be joined
        modified_input.idlist_features = KeyedJaggedTensor.concat(
            [modified_input.idlist_features, self._extra_input.idlist_features]
        )
        if self._extra_input.idscore_features is not None:
            # stride will be smae but features will be joined
            modified_input.idscore_features = KeyedJaggedTensor.concat(
                # pyre-ignore
                [modified_input.idscore_features, self._extra_input.idscore_features]
            )

        # dim=0 (batch dimensions) increases by self._extra_input.input_label.shape[0]
        modified_input.label = torch.concat(
            (modified_input.label, self._extra_input.label), dim=0
        )

        return modified_input

