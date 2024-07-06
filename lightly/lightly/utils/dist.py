def eye_rank(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Returns an (n, n * world_size) zero matrix with the diagonal for the rank
    of this process set to 1.

    Example output where n=3, the current process has rank 1, and there are
    4 processes in total:

        rank0   rank1   rank2   rank3
        0 0 0 | 1 0 0 | 0 0 0 | 0 0 0
        0 0 0 | 0 1 0 | 0 0 0 | 0 0 0
        0 0 0 | 0 0 1 | 0 0 0 | 0 0 0

    Equivalent to torch.eye for undistributed settings or if world size == 1.

    Args:
        n:
            Size of the square matrix on a single process.
        device:
            Device on which the matrix should be created.

    """
    rows = torch.arange(n, device=device, dtype=torch.long)
    cols = rows + rank() * n
    diag_mask = torch.zeros((n, n * world_size()), dtype=torch.bool)
    diag_mask[(rows, cols)] = True
    return diag_mask

