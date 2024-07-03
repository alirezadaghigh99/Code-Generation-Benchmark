def extract_rcrops(
    img: np.ndarray, polys: np.ndarray, dtype=np.float32, channels_last: bool = True
) -> List[np.ndarray]:
    """Created cropped images from list of rotated bounding boxes

    Args:
    ----
        img: input image
        polys: bounding boxes of shape (N, 4, 2)
        dtype: target data type of bounding boxes
        channels_last: whether the channel dimensions is the last one instead of the last one

    Returns:
    -------
        list of cropped images
    """
    if polys.shape[0] == 0:
        return []
    if polys.shape[1:] != (4, 2):
        raise AssertionError("polys are expected to be quadrilateral, of shape (N, 4, 2)")

    # Project relative coordinates
    _boxes = polys.copy()
    height, width = img.shape[:2] if channels_last else img.shape[-2:]
    if not np.issubdtype(_boxes.dtype, np.integer):
        _boxes[:, :, 0] *= width
        _boxes[:, :, 1] *= height

    src_pts = _boxes[:, :3].astype(np.float32)
    # Preserve size
    d1 = np.linalg.norm(src_pts[:, 0] - src_pts[:, 1], axis=-1)
    d2 = np.linalg.norm(src_pts[:, 1] - src_pts[:, 2], axis=-1)
    # (N, 3, 2)
    dst_pts = np.zeros((_boxes.shape[0], 3, 2), dtype=dtype)
    dst_pts[:, 1, 0] = dst_pts[:, 2, 0] = d1 - 1
    dst_pts[:, 2, 1] = d2 - 1
    # Use a warp transformation to extract the crop
    crops = [
        cv2.warpAffine(
            img if channels_last else img.transpose(1, 2, 0),
            # Transformation matrix
            cv2.getAffineTransform(src_pts[idx], dst_pts[idx]),
            (int(d1[idx]), int(d2[idx])),
        )
        for idx in range(_boxes.shape[0])
    ]
    return crops  # type: ignore[return-value]