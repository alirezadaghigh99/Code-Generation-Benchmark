    def from_tensor_list(
        cls,
        tensor_list: list[Tensor],
    ) -> MultiEmbeddingTensor:
        r"""Creates a :class:`MultiEmbeddingTensor` from a list of
        :class:`torch.Tensor`.

        Args:
            tensor_list (List[Tensor]): A list of tensors, where each tensor
                has the same number of rows and can have a different number of
                columns.

        Returns:
            MultiEmbeddingTensor: A :class:`MultiEmbeddingTensor` instance.
        """
        assert isinstance(tensor_list, list) and len(tensor_list) > 0
        num_rows = tensor_list[0].size(0)
        device = tensor_list[0].device
        for tensor in tensor_list:
            msg = "tensor_list must be a list of tensors."
            assert isinstance(tensor, torch.Tensor), msg
            msg = "tensor_list must be a list of 2D tensors."
            assert tensor.dim() == 2, msg
            msg = "num_rows must be the same across a list of input tensors."
            assert tensor.size(0) == num_rows, msg
            msg = "device must be the same across a list of input tensors."
            assert tensor.device == device, msg

        offset_list = []
        accum_idx = 0
        offset_list.append(accum_idx)
        for tensor in tensor_list:
            accum_idx += tensor.size(1)
            offset_list.append(accum_idx)

        num_cols = len(tensor_list)
        values = torch.cat(tensor_list, dim=1)
        offset = torch.tensor(offset_list, device=values.device)
        return cls(num_rows, num_cols, values, offset)