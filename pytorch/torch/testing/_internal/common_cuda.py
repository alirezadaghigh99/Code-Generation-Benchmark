def _get_torch_cuda_version():
    if torch.version.cuda is None:
        return (0, 0)
    cuda_version = str(torch.version.cuda)
    return tuple(int(x) for x in cuda_version.split("."))

def _create_scaling_case(device="cuda", dtype=torch.float, optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
    data = [(torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device))]

    loss_fn = torch.nn.MSELoss().to(device)

    skip_iter = 2

    return _create_scaling_models_optimizers(
        device=device, optimizer_ctor=optimizer_ctor, optimizer_kwargs=optimizer_kwargs,
    ) + (data, loss_fn, skip_iter)

def tf32_off():
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul

