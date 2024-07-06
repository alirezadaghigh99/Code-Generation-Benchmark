def zca_mean(
    inp: Tensor, dim: int = 0, unbiased: bool = True, eps: float = 1e-6, return_inverse: bool = False
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    r"""Compute the ZCA whitening matrix and mean vector.

    The output can be used with :py:meth:`~kornia.color.linear_transform`.
    See :class:`~kornia.color.ZCAWhitening` for details.

    Args:
        inp: input data tensor.
        dim: Specifies the dimension that serves as the samples dimension.
        unbiased: Whether to use the unbiased estimate of the covariance matrix.
        eps: a small number used for numerical stability.
        return_inverse: Whether to return the inverse ZCA transform.

    Shapes:
        - inp: :math:`(D_0,...,D_{\text{dim}},...,D_N)` is a batch of N-D tensors.
        - transform_matrix: :math:`(\Pi_{d=0,d\neq \text{dim}}^N D_d, \Pi_{d=0,d\neq \text{dim}}^N D_d)`
        - mean_vector: :math:`(1, \Pi_{d=0,d\neq \text{dim}}^N D_d)`
        - inv_transform: same shape as the transform matrix

    Returns:
        A tuple containing the ZCA matrix and the mean vector. If return_inverse is set to True,
        then it returns the inverse ZCA matrix, otherwise it returns None.

    .. note::
       See a working example `here <https://colab.sandbox.google.com/github/kornia/tutorials/
       blob/master/source/zca_whitening.ipynb>`__.

    Examples:
        >>> x = torch.tensor([[0,1],[1,0],[-1,0],[0,-1]], dtype = torch.float32)
        >>> transform_matrix, mean_vector,_ = zca_mean(x) # Returns transformation matrix and data mean
        >>> x = torch.rand(3,20,2,2)
        >>> transform_matrix, mean_vector, inv_transform = zca_mean(x, dim = 1, return_inverse = True)
        >>> # transform_matrix.size() equals (12,12) and the mean vector.size equal (1,12)
    """

    if not isinstance(inp, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(inp)}")

    if not isinstance(eps, float):
        raise TypeError(f"eps type is not a float. Got{type(eps)}")

    if not isinstance(unbiased, bool):
        raise TypeError(f"unbiased type is not bool. Got{type(unbiased)}")

    if not isinstance(dim, int):
        raise TypeError(f"Argument 'dim' must be of type int. Got {type(dim)}")

    if not isinstance(return_inverse, bool):
        raise TypeError(f"Argument return_inverse must be of type bool {type(return_inverse)}")

    inp_size = inp.size()

    if dim >= len(inp_size) or dim < -len(inp_size):
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-len(inp_size)},{len(inp_size) - 1}], but got {dim}"
        )

    if dim < 0:
        dim = len(inp_size) + dim

    feat_dims = concatenate([torch.arange(0, dim), torch.arange(dim + 1, len(inp_size))])

    new_order: List[int] = concatenate([tensor([dim]), feat_dims]).tolist()

    inp_permute = inp.permute(new_order)

    N = inp_size[dim]
    feature_sizes = tensor(inp_size[0:dim] + inp_size[dim + 1 : :])
    num_features: int = int(torch.prod(feature_sizes).item())

    mean: Tensor = torch.mean(inp_permute, dim=0, keepdim=True)

    mean = mean.reshape((1, num_features))

    inp_center_flat: Tensor = inp_permute.reshape((N, num_features)) - mean

    cov = inp_center_flat.t().mm(inp_center_flat)

    if unbiased:
        cov = cov / float(N - 1)
    else:
        cov = cov / float(N)

    U, S, _ = torch.linalg.svd(cov)

    S = S.reshape(-1, 1)
    S_inv_root: Tensor = torch.rsqrt(S + eps)
    T: Tensor = (U).mm(S_inv_root * U.t())

    T_inv: Optional[Tensor] = None
    if return_inverse:
        T_inv = (U).mm(torch.sqrt(S + eps) * U.t())

    return T, mean, T_inv

