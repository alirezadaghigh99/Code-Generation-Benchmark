class ITEPEmbeddingBagCollection(nn.Module):
    """
    ITEPEmbeddingBagCollection represents a EmbeddingBagCollection module and an In-Training Embedding Pruning (ITEP) module.
    The inputs into the ITEP-EBC will first be modified by the ITEP module before being passed into the embedding bag collection.
    Args:
        embedding_bag_collection (EmbeddingBagCollection): The EmbeddingBagCollection module to lookup embeddings.
        itep_module (GenericITEPModule): A single ITEP module that modifies the input features.
    Example:
        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=ebc,
            itep_module=itep_module
        )
    Note:
        The forward method modifies the input features using the ITEP module before passing them to the EmbeddingBagCollection.
        It also increments an internal iteration counter each time it is called.
    For details of input and output types, see EmbeddingBagCollection.
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        itep_module: GenericITEPModule,
    ) -> None:
        super().__init__()
        self._embedding_bag_collection = embedding_bag_collection
        self._itep_module = itep_module
        # Iteration counter for ITEP. Pinning on CPU because used for condition checking and checkpointing.
        self.register_buffer(
            "_iter",
            torch.tensor(0, dtype=torch.int64, device=torch.device("cpu")),
        )

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        """
        Forward pass for the ITEPEmbeddingBagCollection module.
        The input features are first passed through the ITEP module, which modifies them.
        The modified features are then passed to the EmbeddingBagCollection to get the pooled embeddings.
        The internal iteration counter is incremented at each call.
        Args:
            features (KeyedJaggedTensor): The input features for the embedding lookup.
        Returns:
            KeyedTensor: The pooled embeddings from the EmbeddingBagCollection.
        Note:
            The iteration counter is incremented after each forward pass to keep track of the number of iterations.
        """

        features = self._itep_module(features, self._iter.item())
        pooled_embeddings = self._embedding_bag_collection(features)
        self._iter += 1

        return pooled_embeddings

    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_collection.embedding_bag_configs()

