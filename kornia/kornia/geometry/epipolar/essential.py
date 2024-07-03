def decompose_essential_matrix_no_svd(E_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Decompose the essential matrix to rotation and translation.

       Recover rotation and translation from essential matrices without SVD
      reference: Horn, Berthold KP. Recovering baseline and orientation from essential matrix[J].
      J. Opt. Soc. Am, 1990, 110.

      Args:
       E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
       A tuple containing the first and second possible rotation matrices and the translation vector.
       The shape of the tensors with be same input :math:`[(*, 3, 3), (*, 3, 3), (*, 3, 1)]`.
    """
    if not (len(E_mat.shape) >= 2 and E_mat.shape[-2:] == (3, 3)):
        raise AssertionError(E_mat.shape)
    elif len(E_mat.shape) != 3:
        E_mat = E_mat.view(-1, 3, 3)

    B = E_mat.shape[0]

    # Eq.18, choose the largest of the three possible pairwise cross-products
    e1, e2, e3 = E_mat[..., 0], E_mat[..., 1], E_mat[..., 2]

    # sqrt(1/2 trace(EE^T)), B
    scale_factor = torch.sqrt(0.5 * torch.diagonal(E_mat @ E_mat.transpose(-1, -2), dim1=-1, dim2=-2).sum(-1))

    # B, 3, 3
    cross_products = torch.stack([torch.cross(e1, e2), torch.cross(e2, e3), torch.cross(e3, e1)], dim=1)
    # B, 3, 1
    norms = torch.norm(cross_products, dim=-1, keepdim=True)

    # B, to select which b1
    largest = torch.argmax(norms, dim=-2)

    # B, 3, 3
    e_cross_products = scale_factor[:, None, None] * cross_products / norms

    # broadcast the index
    index_expanded = largest.unsqueeze(-1).expand(-1, -1, e_cross_products.size(-1))

    # slice at dim=1, select for each batch one b (e1*e2 or e2*e3 or e3*e1), B, 1, 3
    b1 = torch.gather(e_cross_products, dim=1, index=index_expanded).squeeze(1)
    # normalization
    b1_ = b1 / torch.norm(b1, dim=-1, keepdim=True)

    # skew-symmetric matrix
    B1 = torch.zeros((B, 3, 3), device=E_mat.device, dtype=E_mat.dtype)
    t0, t1, t2 = b1[:, 0], b1[:, 1], b1[:, 2]
    B1[:, 0, 1], B1[:, 1, 0] = -t2, t2
    B1[:, 0, 2], B1[:, 2, 0] = t1, -t1
    B1[:, 1, 2], B1[:, 2, 1] = -t0, t0

    # the second translation and rotation
    B2 = -B1
    b2 = -b1

    # Eq.24, recover R
    # (bb)R = Cofactors(E)^T - BE
    R1 = (matrix_cofactor_tensor(E_mat) - B1 @ E_mat) / (b1 * b1).sum().unsqueeze(-1)
    R2 = (matrix_cofactor_tensor(E_mat) - B2 @ E_mat) / (b2 * b2).sum().unsqueeze(-1)

    return (R1, R2, b1_.unsqueeze(-1))def motion_from_essential_choose_solution(
    E_mat: torch.Tensor,
    K1: torch.Tensor,
    K2: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Recover the relative camera rotation and the translation from an estimated essential matrix.

    The method checks the corresponding points in two images and also returns the triangulated
    3d points. Internally uses :py:meth:`~kornia.geometry.epipolar.decompose_essential_matrix` and then chooses
    the best solution based on the combination that gives more 3d points in front of the camera plane from
    :py:meth:`~kornia.geometry.epipolar.triangulate_points`.

    Args:
        E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.
        K1: The camera matrix from first camera with shape :math:`(*, 3, 3)`.
        K2: The camera matrix from second camera with shape :math:`(*, 3, 3)`.
        x1: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        x2: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        mask: A boolean mask which can be used to exclude some points from choosing
          the best solution. This is useful for using this function with sets of points of
          different cardinality (for instance after filtering with RANSAC) while keeping batch
          semantics. Mask is of shape :math:`(*, N)`.

    Returns:
        The rotation and translation plus the 3d triangulated points.
        The tuple is as following :math:`[(*, 3, 3), (*, 3, 1), (*, N, 3)]`.
    """
    if not (len(E_mat.shape) >= 2 and E_mat.shape[-2:] == (3, 3)):
        raise AssertionError(E_mat.shape)
    if not (len(K1.shape) >= 2 and K1.shape[-2:] == (3, 3)):
        raise AssertionError(K1.shape)
    if not (len(K2.shape) >= 2 and K2.shape[-2:] == (3, 3)):
        raise AssertionError(K2.shape)
    if not (len(x1.shape) >= 2 and x1.shape[-1] == 2):
        raise AssertionError(x1.shape)
    if not (len(x2.shape) >= 2 and x2.shape[-1] == 2):
        raise AssertionError(x2.shape)
    if not len(E_mat.shape[:-2]) == len(K1.shape[:-2]) == len(K2.shape[:-2]):
        raise AssertionError
    if mask is not None:
        if len(mask.shape) < 1:
            raise AssertionError(mask.shape)
        if mask.shape != x1.shape[:-1]:
            raise AssertionError(mask.shape)

    unbatched = len(E_mat.shape) == 2

    if unbatched:
        # add a leading batch dimension. We will remove it at the end, before
        # returning the results
        E_mat = E_mat[None]
        K1 = K1[None]
        K2 = K2[None]
        x1 = x1[None]
        x2 = x2[None]
        if mask is not None:
            mask = mask[None]

    # compute four possible pose solutions
    Rs, ts = motion_from_essential(E_mat)

    # set reference view pose and compute projection matrix
    R1 = eye_like(3, E_mat)  # Bx3x3
    t1 = vec_like(3, E_mat)  # Bx3x1

    # compute the projection matrices for first camera
    R1 = R1[:, None].expand(-1, 4, -1, -1)
    t1 = t1[:, None].expand(-1, 4, -1, -1)
    K1 = K1[:, None].expand(-1, 4, -1, -1)
    P1 = projection_from_KRt(K1, R1, t1)  # 1x4x4x4

    # compute the projection matrices for second camera
    R2 = Rs
    t2 = ts
    K2 = K2[:, None].expand(-1, 4, -1, -1)
    P2 = projection_from_KRt(K2, R2, t2)  # Bx4x4x4

    # triangulate the points
    x1 = x1[:, None].expand(-1, 4, -1, -1)
    x2 = x2[:, None].expand(-1, 4, -1, -1)
    X = triangulate_points(P1, P2, x1, x2)  # Bx4xNx3

    # project points and compute their depth values
    d1 = depth_from_point(R1, t1, X)
    d2 = depth_from_point(R2, t2, X)

    # verify the point values that have a positive depth value
    depth_mask = (d1 > 0.0) & (d2 > 0.0)
    if mask is not None:
        depth_mask &= mask.unsqueeze(1)

    mask_indices = torch.max(depth_mask.sum(-1), dim=-1, keepdim=True)[1]

    # get pose and points 3d and return
    R_out = Rs[:, mask_indices][:, 0, 0]
    t_out = ts[:, mask_indices][:, 0, 0]
    points3d_out = X[:, mask_indices][:, 0, 0]

    if unbatched:
        R_out = R_out[0]
        t_out = t_out[0]
        points3d_out = points3d_out[0]

    return R_out, t_out, points3d_outdef motion_from_essential(E_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Get Motion (R's and t's ) from Essential matrix.

    Computes and return four possible poses exist for the decomposition of the Essential
    matrix. The possible solutions are :math:`[R1,t], [R1,-t], [R2,t], [R2,-t]`.

    Args:
        E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
        The rotation and translation containing the four possible combination for the retrieved motion.
        The tuple is as following :math:`[(*, 4, 3, 3), (*, 4, 3, 1)]`.
    """
    if not (len(E_mat.shape) >= 2 and E_mat.shape[-2:] == (3, 3)):
        raise AssertionError(E_mat.shape)

    # decompose the essential matrix by its possible poses
    R1, R2, t = decompose_essential_matrix(E_mat)

    # compbine and returns the four possible solutions
    Rs = stack([R1, R1, R2, R2], dim=-3)
    Ts = stack([t, -t, t, -t], dim=-3)

    return Rs, Ts