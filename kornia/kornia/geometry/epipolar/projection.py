def scale_intrinsics(camera_matrix: Tensor, scale_factor: Union[float, Tensor]) -> Tensor:
    r"""Scale a camera matrix containing the intrinsics.

    Applies the scaling factor to the focal length and center of projection.

    Args:
        camera_matrix: the camera calibration matrix containing the intrinsic
          parameters. The expected shape for the tensor is :math:`(B, 3, 3)`.
        scale_factor: the scaling factor to be applied.

    Returns:
        The scaled camera matrix with shame shape as input :math:`(B, 3, 3)`.
    """
    K_scale = camera_matrix.clone()
    K_scale[..., 0, 0] *= scale_factor
    K_scale[..., 1, 1] *= scale_factor
    K_scale[..., 0, 2] *= scale_factor
    K_scale[..., 1, 2] *= scale_factor
    return K_scale

