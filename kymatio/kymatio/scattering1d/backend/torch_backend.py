def rfft(cls, x):
        cls.contiguous_check(x)
        cls.real_check(x)

        x_r = torch.zeros(
            x.shape[:-1] + (2,), dtype=x.dtype, layout=x.layout, device=x.device
        )
        x_r[..., 0] = x[..., 0]

        return _fft(x_r)

