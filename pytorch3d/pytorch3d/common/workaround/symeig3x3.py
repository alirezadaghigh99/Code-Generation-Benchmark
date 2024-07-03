def symeig3x3(
    inputs: torch.Tensor, eigenvectors: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute eigenvalues and (optionally) eigenvectors

    Args:
        inputs: symmetric matrices with shape of (..., 3, 3)
        eigenvectors: whether should we compute only eigenvalues or eigenvectors as well

    Returns:
        Either a tuple of (eigenvalues, eigenvectors) or eigenvalues only, depending on
         given params. Eigenvalues are of shape (..., 3) and eigenvectors (..., 3, 3)
    """
    return _SymEig3x3().to(inputs.device)(inputs, eigenvectors=eigenvectors)