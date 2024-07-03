def match_fginn(
    desc1: Tensor,
    desc2: Tensor,
    lafs1: Tensor,
    lafs2: Tensor,
    th: float = 0.8,
    spatial_th: float = 10.0,
    mutual: bool = False,
    dm: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Function, which finds nearest neighbors in desc2 for each vector in desc1.

    The method satisfies first to second nearest neighbor distance <= th,
    and assures 2nd nearest neighbor is geometrically inconsistent with the 1st one
    (see :cite:`MODS2015` for more details)

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        lafs1: LAFs of a shape :math:`(1, B1, 2, 3)`.
        lafs2: LAFs of a shape :math:`(1, B2, 2, 3)`.

        th: distance ratio threshold.
        spatial_th: minimal distance in pixels to 2nd nearest neighbor.
        mutual: also perform mutual nearest neighbor check
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Return:
        - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
        - Long tensor indexes of matching descriptors in desc1 and desc2. Shape: :math:`(B3, 2)`,
          where 0 <= B3 <= B1.
    """
    KORNIA_CHECK_SHAPE(desc1, ["B", "DIM"])
    KORNIA_CHECK_SHAPE(desc2, ["B", "DIM"])
    BIG_NUMBER = 1000000.0

    distance_matrix = _get_lazy_distance_matrix(desc1, desc2, dm)
    dtype = distance_matrix.dtype

    if desc2.shape[0] < 2:  # We cannot perform snn check, so output empty matches
        return _no_match(distance_matrix)

    num_candidates = max(2, min(10, desc2.shape[0]))
    vals_cand, idxs_in_2 = torch.topk(distance_matrix, num_candidates, dim=1, largest=False)
    vals = vals_cand[:, 0]
    xy2 = get_laf_center(lafs2).view(-1, 2)
    candidates_xy = xy2[idxs_in_2]
    kdist = torch.norm(candidates_xy - candidates_xy[0:1], p=2, dim=2)
    fginn_vals = vals_cand[:, 1:] + (kdist[:, 1:] < spatial_th).to(dtype) * BIG_NUMBER
    fginn_vals_best, fginn_idxs_best = fginn_vals.min(dim=1)

    # orig_idxs = idxs_in_2.gather(1, fginn_idxs_best.unsqueeze(1))[0]
    # if you need to know fginn indexes - uncomment

    vals_2nd = fginn_vals_best
    idxs_in_2 = idxs_in_2[:, 0]

    ratio = vals / vals_2nd
    mask = ratio <= th
    match_dists = ratio[mask]
    if len(match_dists) == 0:
        return _no_match(distance_matrix)
    idxs_in1 = torch.arange(0, idxs_in_2.size(0), device=distance_matrix.device)[mask]
    idxs_in_2 = idxs_in_2[mask]
    matches_idxs = concatenate([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], 1)
    match_dists, matches_idxs = match_dists.view(-1, 1), matches_idxs.view(-1, 2)

    if not mutual:  # returning 1-way matches
        return match_dists, matches_idxs
    _, idxs_in_1_mut = torch.min(distance_matrix, dim=0)
    good_mask = matches_idxs[:, 0] == idxs_in_1_mut[matches_idxs[:, 1]]
    return match_dists[good_mask], matches_idxs[good_mask]