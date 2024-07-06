def from_torch_tensor(tensor: torch.Tensor, tensor_meta: Optional[TensorMeta] = None) -> "TracedTensor":
        """
        Creates a TracedTensor by patching a given torch.Tensor, associating it with the provided tensor_meta.

        :param tensor: The input torch.Tensor.
        :param tensor_meta: The metadata associated with the tensor.
        :return: The resulting TracedTensor.
        """
        return TracedTensor.patch(tensor, tensor_meta)

