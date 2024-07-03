        def randperm(x: int):
            perm = torch.randperm(x)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert perm.dtype == torch.int64