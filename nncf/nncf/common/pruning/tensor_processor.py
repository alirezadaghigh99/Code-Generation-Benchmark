def concatenate(cls, tensors: List[NNCFTensor], axis: int) -> NNCFTensor:
        """
        Join a list of NNCFTensors along an existing axis.

        :param tensors: List of NNCFTensors.
        :param axis: The axis, along which the tensors will be joined.
        :returns: The concatenated List of the tensors.
        """

