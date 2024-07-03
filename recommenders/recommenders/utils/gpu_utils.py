def get_number_gpus():
    """Get the number of GPUs in the system.
    Returns:
        int: Number of GPUs.
    """
    try:
        import torch

        return torch.cuda.device_count()
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        import numba

        return len(numba.cuda.gpus)
    except Exception:  # numba.cuda.cudadrv.error.CudaSupportError:
        return 0