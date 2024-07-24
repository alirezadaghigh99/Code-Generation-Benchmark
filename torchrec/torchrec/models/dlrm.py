class InteractionDCNArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the output of a Deep Cross Net v2
    https://arxiv.org/pdf/2008.13535.pdf with a low rank approximation for the
    weight matrix. The input and output sizes are the same for this
    interaction layer (F*D + D).

    .. note::
        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them
        can be computed.


    Args:
        num_sparse_features (int): F.

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        DCN = LowRankCrossNet(
            in_features = F*D+D,
            dcn_num_layers = 2,
            dnc_low_rank_dim = 4,
        )
        inter_arch = InteractionDCNArch(
            num_sparse_features=len(keys),
            crossnet=DCN,
        )

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        #  B X (F*D + D)
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(self, num_sparse_features: int, crossnet: nn.Module) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.crossnet = crossnet

    def forward(
        self, dense_features: torch.Tensor, sparse_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.

        Returns:
            torch.Tensor: an output tensor of size B X (F*D + D).
        """
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        combined_values = torch.cat(
            (dense_features.unsqueeze(1), sparse_features), dim=1
        )

        # size B X (F*D + D)
        return self.crossnet(combined_values.reshape([B, -1]))

class InteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the pairwise dot product of each sparse feature pair,
    the dot product of each sparse features with the output of the dense layer,
    and the dense layer itself (all concatenated).

    .. note::
        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them
        can be computed.


    Args:
        num_sparse_features (int): F.

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=len(keys))

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        #  B X (D + F + F choose 2)
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(self, num_sparse_features: int) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.register_buffer(
            "triu_indices",
            torch.triu_indices(self.F + 1, self.F + 1, offset=1),
            persistent=False,
        )

    def forward(
        self, dense_features: torch.Tensor, sparse_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.

        Returns:
            torch.Tensor: an output tensor of size B X (D + F + F choose 2).
        """
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        combined_values = torch.cat(
            (dense_features.unsqueeze(1), sparse_features), dim=1
        )

        # dense/sparse + sparse/sparse interaction
        # size B X (F + F choose 2)
        interactions = torch.bmm(
            combined_values, torch.transpose(combined_values, 1, 2)
        )
        interactions_flat = interactions[:, self.triu_indices[0], self.triu_indices[1]]

        return torch.cat((dense_features, interactions_flat), dim=1)

