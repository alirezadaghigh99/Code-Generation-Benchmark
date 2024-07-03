def cosine_basis_functions(x, n_basis_functions=64):
    """Cosine basis functions used to embed quantile thresholds.

    Args:
        x (torch.Tensor): Input.
        n_basis_functions (int): Number of cosine basis functions.

    Returns:
        ndarray: Embedding with shape of (x.shape + (n_basis_functions,)).
    """
    # Equation (4) in the IQN paper has an error stating i=0,...,n-1.
    # Actually i=1,...,n is correct (personal communication)
    i_pi = (
        torch.arange(1, n_basis_functions + 1, dtype=torch.float, device=x.device)
        * np.pi
    )
    embedding = torch.cos(x[..., None] * i_pi)
    assert embedding.shape == x.shape + (n_basis_functions,)
    return embedding