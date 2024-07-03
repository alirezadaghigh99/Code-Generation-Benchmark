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
    return x_min, 1 - y_max, x_max, 1 - y_mindef d4(img: np.ndarray, group_member: D4Type) -> np.ndarray:
    """Applies a `D_4` symmetry group transformation to an image array.

    This function manipulates an image using transformations such as rotations and flips,
    corresponding to the `D_4` dihedral group symmetry operations.
    Each transformation is identified by a unique group member code.

    Parameters:
    - img (np.ndarray): The input image array to transform.
    - group_member (D4Type): A string identifier indicating the specific transformation to apply. Valid codes include:
      - 'e': Identity (no transformation).
      - 'r90': Rotate 90 degrees counterclockwise.
      - 'r180': Rotate 180 degrees.
      - 'r270': Rotate 270 degrees counterclockwise.
      - 'v': Vertical flip.
      - 'hvt': Transpose over second diagonal
      - 'h': Horizontal flip.
      - 't': Transpose (reflect over the main diagonal).

    Returns:
    - np.ndarray: The transformed image array.

    Raises:
    - ValueError: If an invalid group member is specified.

    Examples:
    - Rotating an image by 90 degrees:
      `transformed_image = d4(original_image, 'r90')`
    - Applying a horizontal flip to an image:
      `transformed_image = d4(original_image, 'h')`
    """
    transformations = {
        "e": lambda x: x,  # Identity transformation
        "r90": lambda x: rot90(x, 1),  # Rotate 90 degrees
        "r180": lambda x: rot90(x, 2),  # Rotate 180 degrees
        "r270": lambda x: rot90(x, 3),  # Rotate 270 degrees
        "v": vflip,  # Vertical flip
        "hvt": lambda x: transpose(rot90(x, 2)),  # Reflect over anti-diagonal
        "h": hflip,  # Horizontal flip
        "t": transpose,  # Transpose (reflect over main diagonal)
    }

    # Execute the appropriate transformation
    if group_member in transformations:
        return np.ascontiguousarray(transformations[group_member](img))

    raise ValueError(f"Invalid group member: {group_member}")def warp_affine(
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
    return warp_fn(image)