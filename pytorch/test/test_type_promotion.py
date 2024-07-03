        def make_tensor(size, dtype):
            if dtype == torch.bool:
                return torch.randint(2, size, dtype=dtype, device=device)
            elif dtype == torch.int:
                return torch.randint(10, size, dtype=dtype, device=device)
            else:
                return torch.randn(size, dtype=dtype, device=device)        def make_tensor(size, dtype):
            if dtype == torch.bool:
                return torch.randint(2, size, dtype=dtype, device=device)
            elif dtype == torch.int:
                return torch.randint(10, size, dtype=dtype, device=device)
            else:
                return torch.randn(size, dtype=dtype, device=device)