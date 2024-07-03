def warp_affine(
    image: np.ndarray,
    matrix: skimage.transform.ProjectiveTransform,
    interpolation: int,
    cval: ColorType,
    mode: int,
    output_shape: Sequence[int],
) -> np.ndarray:
    if _is_identity_matrix(matrix):
        return image

    dsize = int(np.round(output_shape[1])), int(np.round(output_shape[0]))
    warp_fn = maybe_process_in_chunks(
        cv2.warpAffine,
        M=matrix.params[:2],
        dsize=dsize,
        flags=interpolation,
        borderMode=mode,
        borderValue=cval,
    )
    return warp_fn(image)def vflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ...])def bbox_vflip(bbox: BoxInternalType, rows: int | None = None, cols: int | None = None) -> BoxInternalType:
    """Flip a bounding box vertically around the x-axis.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image rows.
        cols: Image cols.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    return x_min, 1 - y_max, x_max, 1 - y_min