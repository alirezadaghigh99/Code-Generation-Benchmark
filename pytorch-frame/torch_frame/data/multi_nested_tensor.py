def from_tensor_mat(
        cls,
        tensor_mat: list[list[Tensor]],
    ) -> MultiNestedTensor:
        r"""Construct :class:`MultiNestedTensor` object from
        :obj:`tensor_mat`.

        Args:
            tensor_mat (List[List[Tensor]]): A matrix of
                :class:`torch.Tensor` objects. :obj:`tensor_mat[i][j]`
                contains 1-dim :class:`torch.Tensor` of :obj:`i`-th row
                and :obj:`j`-th column, varying in size.

        Returns:
            MultiNestedTensor: A :class:`MultiNestedTensor` instance.
        """
        num_rows = len(tensor_mat)
        num_cols = len(tensor_mat[0])

        offset_list = []
        accum_idx = 0
        offset_list.append(accum_idx)
        values_list = []
        for i in range(num_rows):
            if len(tensor_mat[i]) != num_cols:
                raise RuntimeError(
                    f"The length of each row must be the same."
                    f" tensor_mat[0] has length {num_cols}, but"
                    f" tensor_mat[{i}] has length {len(tensor_mat[i])}")

            for j in range(num_cols):
                tensor = tensor_mat[i][j]
                if not isinstance(tensor, Tensor):
                    raise RuntimeError(
                        "The element of tensor_mat must be PyTorch Tensor")
                if tensor.ndim != 1:
                    raise RuntimeError(
                        "tensor in tensor_mat needs to be 1-dimensional.")
                values_list.append(tensor)
                accum_idx += len(tensor)
                offset_list.append(accum_idx)

        values = torch.cat(values_list)
        offset = torch.tensor(offset_list, device=values.device)

        return cls(num_rows, num_cols, values, offset)

