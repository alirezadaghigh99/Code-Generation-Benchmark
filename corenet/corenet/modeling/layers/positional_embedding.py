class PositionalEmbedding(BaseLayer):
    def __init__(
        self,
        opts,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        is_learnable: Optional[bool] = False,
        sequence_first: Optional[bool] = False,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        module = (
            LearnablePositionalEmbedding
            if is_learnable
            else SinusoidalPositionalEmbedding
        )
        self.pos_embed = module(
            opts,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            sequence_first=sequence_first,
            interpolation_mode=interpolation_mode,
            *args,
            **kwargs
        )

    def forward(self, seq_len: int, *args, **kwargs) -> Tensor:
        return self.pos_embed(seq_len, *args, **kwargs)

    def __repr__(self):
        return self.pos_embed.__repr__()

