class TokenEmbedding(nn.Module):
    """The token embedding class. Converts tensor of input indices into corresponding tensor of token embeddings."""

    def __init__(self, vocab_size: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        """Forward pass through token embedding module.
        :param tokens: Tokens to embed
        :type tokens: torch.Tensor
        """
        # return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        return self.embedding(tokens)