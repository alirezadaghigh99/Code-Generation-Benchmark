class VariableBatchPooledEmbeddingsAllToAll(nn.Module):
    """
    Shards batches and collects keys of tensor with a `ProcessGroup` according to
    `dim_sum_per_rank`.

    Implementation utilizes `variable_batch_alltoall_pooled` operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        emb_dim_per_rank_per_feature (List[List[int]]): embedding dimensions per rank
            per feature.
        device (Optional[torch.device]): device on which buffers will be allocated.
        callbacks (Optional[List[Callable[[torch.Tensor], torch.Tensor]]]): callback
            functions.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.

    Example::

        kjt_split = [1, 2]
        emb_dim_per_rank_per_feature = [[2], [3, 3]]
        a2a = VariableBatchPooledEmbeddingsAllToAll(
            pg, emb_dim_per_rank_per_feature, device
        )

        t0 = torch.rand(6) # 2 * (2 + 1)
        t1 = torch.rand(24) # 3 * (1 + 3) + 3 * (2 + 2)
        #        r0_batch_size   r1_batch_size
        #  f_0:              2               1
        -----------------------------------------
        #  f_1:              1               2
        #  f_2:              3               2
        r0_batch_size_per_rank_per_feature = [[2], [1]]
        r1_batch_size_per_rank_per_feature = [[1, 3], [2, 2]]
        r0_batch_size_per_feature_pre_a2a = [2, 1, 3]
        r1_batch_size_per_feature_pre_a2a = [1, 2, 2]

        rank0_output = a2a(
            t0, r0_batch_size_per_rank_per_feature, r0_batch_size_per_feature_pre_a2a
        ).wait()
        rank1_output = a2a(
            t1, r1_batch_size_per_rank_per_feature, r1_batch_size_per_feature_pre_a2a
        ).wait()

        # input splits:
        #   r0: [2*2, 1*2]
        #   r1: [1*3 + 3*3, 2*3 + 2*3]

        # output splits:
        #   r0: [2*2, 1*3 + 3*3]
        #   r1: [1*2, 2*3 + 2*3]

        print(rank0_output.size())
            # torch.Size([16])
            # 2*2 + 1*3 + 3*3
        print(rank1_output.size())
            # torch.Size([14])
            # 1*2 + 2*3 + 2*3
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        emb_dim_per_rank_per_feature: List[List[int]],
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        codecs: Optional[QuantizedCommCodecs] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once("torchrec.distributed.vbe")
        self._pg = pg
        self._emb_dim_per_rank_per_feature = emb_dim_per_rank_per_feature
        self._callbacks: List[Callable[[torch.Tensor], torch.Tensor]] = []
        if callbacks is not None:
            self._callbacks = callbacks
        self._codecs = codecs

    def forward(
        self,
        local_embs: torch.Tensor,
        batch_size_per_rank_per_feature: List[List[int]],
        batch_size_per_feature_pre_a2a: List[int],
    ) -> PooledEmbeddingsAwaitable:
        """
        Performs AlltoAll pooled operation with variable batch size per feature on a
        pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of values to distribute.
            batch_size_per_rank_per_feature (List[List[int]]): batch size per rank per
                feature, post a2a. Used to get the input splits.
            batch_size_per_feature_pre_a2a (List[int]): local batch size before
                scattering, used to get the output splits.
                Ordered by rank_0 feature, rank_1 feature, ...

        Returns:
            PooledEmbeddingsAwaitable: awaitable of pooled embeddings.
        """

        tensor_awaitable = variable_batch_alltoall_pooled(
            a2a_pooled_embs_tensor=local_embs,
            batch_size_per_rank_per_feature=batch_size_per_rank_per_feature,
            batch_size_per_feature_pre_a2a=batch_size_per_feature_pre_a2a,
            emb_dim_per_rank_per_feature=self._emb_dim_per_rank_per_feature,
            group=self._pg,
            codecs=self._codecs,
        )

        pooled_embedding_awaitable = PooledEmbeddingsAwaitable(
            tensor_awaitable=tensor_awaitable,
        )
        pooled_embedding_awaitable.callbacks.extend(self._callbacks)

        return pooled_embedding_awaitable

    @property
    def callbacks(self) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        return self._callbacks

