def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.
        single_directional: If False (default), loss comes from both the distance between
            each point in x and its nearest neighbor in y and each point in y and its nearest
            neighbor in x. If True, loss is the distance between each point in x and its
            nearest neighbor in y.
        abs_cosine: If False, loss_normals is from one minus the cosine similarity.
            If True (default), loss_normals is from one minus the absolute value of the
            cosine similarity, which means that exactly opposite normals are considered
            equivalent to exactly matching normals, i.e. sign does not matter.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")
    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    cham_x, cham_norm_x = _chamfer_distance_single_direction(
        x,
        y,
        x_lengths,
        y_lengths,
        x_normals,
        y_normals,
        weights,
        batch_reduction,
        point_reduction,
        norm,
        abs_cosine,
    )
    if single_directional:
        return cham_x, cham_norm_x
    else:
        cham_y, cham_norm_y = _chamfer_distance_single_direction(
            y,
            x,
            y_lengths,
            x_lengths,
            y_normals,
            x_normals,
            weights,
            batch_reduction,
            point_reduction,
            norm,
            abs_cosine,
        )
        if point_reduction is not None:
            return (
                cham_x + cham_y,
                (cham_norm_x + cham_norm_y) if cham_norm_x is not None else None,
            )
        return (
            (cham_x, cham_y),
            (cham_norm_x, cham_norm_y) if cham_norm_x is not None else None,
        )

