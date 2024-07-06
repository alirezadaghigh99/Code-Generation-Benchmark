def extract_patches_from_pyramid(
    img: Tensor, laf: Tensor, PS: int = 32, normalize_lafs_before_extraction: bool = True
) -> Tensor:
    """Extract patches defined by LAFs from image tensor.

    Patches are extracted from appropriate pyramid level.

    Args:
        img: images, LAFs are detected in  :math:`(B, CH, H, W)`.
        laf: :math:`(B, N, 2, 3)`.
        PS: patch size.
        normalize_lafs_before_extraction: if True, lafs are normalized to image size.

    Returns:
        patches with shape :math:`(B, N, CH, PS,PS)`.
    """
    KORNIA_CHECK_LAF(laf)
    if normalize_lafs_before_extraction:
        nlaf = normalize_laf(laf, img)
    else:
        nlaf = laf
    B, N, _, _ = laf.size()
    _, ch, h, w = img.size()
    scale = 2.0 * get_laf_scale(denormalize_laf(nlaf, img)) / float(PS)
    max_level = min(img.size(2), img.size(3)) // PS
    pyr_idx = scale.log2().clamp(min=0.0, max=max(0, max_level - 1)).long()
    cur_img = img
    cur_pyr_level = 0
    out = torch.zeros(B, N, ch, PS, PS).to(nlaf.dtype).to(nlaf.device)
    we_are_in_business = True
    while we_are_in_business:
        _, ch, h, w = cur_img.size()
        # for loop temporarily, to be refactored
        for i in range(B):
            scale_mask = (pyr_idx[i] == cur_pyr_level).squeeze()
            if (scale_mask.float().sum().item()) == 0:
                continue
            scale_mask = (scale_mask > 0).view(-1)
            grid = generate_patch_grid_from_normalized_LAF(cur_img[i : i + 1], nlaf[i : i + 1, scale_mask, :, :], PS)
            patches = F.grid_sample(
                cur_img[i : i + 1].expand(grid.shape[0], ch, h, w), grid, padding_mode="border", align_corners=False
            )
            out[i].masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches)
        we_are_in_business = min(cur_img.size(2), cur_img.size(3)) >= PS
        if not we_are_in_business:
            break
        cur_img = pyrdown(cur_img)
        cur_pyr_level += 1
    return out

def make_upright(laf: Tensor, eps: float = 1e-9) -> Tensor:
    """Rectify the affine matrix, so that it becomes upright.

    Args:
        laf: :math:`(B, N, 2, 3)`
        eps: for safe division.

    Returns:
        laf: :math:`(B, N, 2, 3)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = make_upright(input)  #  BxNx2x3
    """
    KORNIA_CHECK_LAF(laf)
    det = get_laf_scale(laf)
    scale = det
    # The function is equivalent to doing 2x2 SVD and resetting rotation
    # matrix to an identity: U, S, V = svd(LAF); LAF_upright = U * S.
    b2a2 = torch.sqrt(laf[..., 0:1, 1:2] ** 2 + laf[..., 0:1, 0:1] ** 2) + eps
    laf1_ell = concatenate([(b2a2 / det).contiguous(), torch.zeros_like(det)], dim=3)
    laf2_ell = concatenate(
        [
            ((laf[..., 1:2, 1:2] * laf[..., 0:1, 1:2] + laf[..., 1:2, 0:1] * laf[..., 0:1, 0:1]) / (b2a2 * det)),
            (det / b2a2).contiguous(),
        ],
        dim=3,
    )
    laf_unit_scale = concatenate([concatenate([laf1_ell, laf2_ell], dim=2), laf[..., :, 2:3]], dim=3)
    return scale_laf(laf_unit_scale, scale)

def set_laf_orientation(LAF: Tensor, angles_degrees: Tensor) -> Tensor:
    """Change the orientation of the LAFs.

    Args:
        LAF: :math:`(B, N, 2, 3)`
        angles: :math:`(B, N, 1)` in degrees.

    Returns:
        LAF oriented with angles :math:`(B, N, 2, 3)`
    """
    KORNIA_CHECK_LAF(LAF)
    B, N = LAF.shape[:2]
    ori = get_laf_orientation(LAF).reshape_as(angles_degrees)
    return rotate_laf(LAF, angles_degrees - ori)

def laf_from_center_scale_ori(xy: Tensor, scale: Optional[Tensor] = None, ori: Optional[Tensor] = None) -> Tensor:
    """Creates a LAF from keypoint center, scale and orientation.

    Useful to create kornia LAFs from OpenCV keypoints.

    Args:
        xy: :math:`(B, N, 2)`.
        scale: :math:`(B, N, 1, 1)`. If not provided, scale = 1.0 is assumed
        angle in degrees: :math:`(B, N, 1)`. If not provided orientation = 0 is assumed

    Returns:
        LAF :math:`(B, N, 2, 3)`
    """
    KORNIA_CHECK_SHAPE(xy, ["B", "N", "2"])
    device = xy.device
    dtype = xy.dtype
    B, N = xy.shape[:2]
    if scale is None:
        scale = torch.ones(B, N, 1, 1, device=device, dtype=dtype)
    if ori is None:
        ori = zeros(B, N, 1, device=device, dtype=dtype)
    KORNIA_CHECK_SHAPE(scale, ["B", "N", "1", "1"])
    KORNIA_CHECK_SHAPE(ori, ["B", "N", "1"])
    unscaled_laf = concatenate([angle_to_rotation_matrix(ori.squeeze(-1)), xy.unsqueeze(-1)], dim=-1)
    laf = scale_laf(unscaled_laf, scale)
    return laf

def laf_from_three_points(threepts: Tensor) -> Tensor:
    """Convert three points to local affine frame.

    Order is (0,0), (0, 1), (1, 0).

    Args:
        threepts: :math:`(B, N, 2, 3)`.

    Returns:
        laf :math:`(B, N, 2, 3)`.
    """
    laf = stack([threepts[..., 0] - threepts[..., 2], threepts[..., 1] - threepts[..., 2], threepts[..., 2]], dim=-1)
    return laf

