def randn(self, *args, **kwargs):
        """
        Variant of torch.randn that also works in the TEST_CUDA case.
        """
        # TODO: Put this in torch.cuda.randn
        return torch.empty(*args, **kwargs).normal_()

def to_dense(sparse, fill_value=None):
            """
            Return dense tensor from a sparse tensor using given fill value.
            """
            if fill_value is None or fill_value == 0:
                return sparse.to_dense()
            sparse = sparse.coalesce()
            dense = torch.full(sparse.shape, fill_value, dtype=sparse.dtype, device=sparse.device)
            for idx, value in zip(sparse._indices().t(), sparse._values()):
                dense[tuple(idx)] = value
            return dense

