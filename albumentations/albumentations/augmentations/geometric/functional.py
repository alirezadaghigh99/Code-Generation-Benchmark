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
    return warp_fn(image)

def vflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ...])

def bbox_vflip(bbox: BoxInternalType, rows: int | None = None, cols: int | None = None) -> BoxInternalType:
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

def d4(img: np.ndarray, group_member: D4Type) -> np.ndarray:
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

    raise ValueError(f"Invalid group member: {group_member}")

def rot90(img: np.ndarray, factor: int) -> np.ndarray:
    return np.rot90(img, factor)

def rotate(
    img: np.ndarray,
    angle: float,
    interpolation: int,
    border_mode: int,
    value: ColorType | None = None,
) -> np.ndarray:
    height, width = img.shape[:2]

    image_center = center(width, height)
    matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    warp_fn = maybe_process_in_chunks(
        cv2.warpAffine,
        M=matrix,
        dsize=(width, height),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    return warp_fn(img)

def keypoint_rot90(
    keypoint: KeypointInternalType,
    factor: int,
    rows: int,
    cols: int,
    **params: Any,
) -> KeypointInternalType:
    """Rotate a keypoint by 90 degrees counter-clockwise (CCW) a specified number of times.

    Args:
        keypoint (KeypointInternalType): A keypoint in the format `(x, y, angle, scale)`.
        factor (int): The number of 90 degree CCW rotations to apply. Must be in the range [0, 3].
        rows (int): The height of the image the keypoint belongs to.
        cols (int): The width of the image the keypoint belongs to.
        **params: Additional parameters.

    Returns:
        KeypointInternalType: The rotated keypoint in the format `(x, y, angle, scale)`.

    Raises:
        ValueError: If the factor is not in the set {0, 1, 2, 3}.
    """
    x, y, angle, scale = keypoint

    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter factor must be in set {0, 1, 2, 3}")

    if factor == 1:
        x, y, angle = y, (cols - 1) - x, angle - math.pi / 2
    elif factor == ROT90_180_FACTOR:
        x, y, angle = (cols - 1) - x, (rows - 1) - y, angle - math.pi
    elif factor == ROT90_270_FACTOR:
        x, y, angle = (rows - 1) - y, x, angle + math.pi / 2

    return x, y, angle, scale

def transpose(img: np.ndarray) -> np.ndarray:
    """Transposes the first two dimensions of an array of any dimensionality.
    Retains the order of any additional dimensions.

    Args:
        img (np.ndarray): Input array.

    Returns:
        np.ndarray: Transposed array.
    """
    # Generate the new axes order
    new_axes = list(range(img.ndim))
    new_axes[0], new_axes[1] = 1, 0  # Swap the first two dimensions

    # Transpose the array using the new axes order
    return img.transpose(new_axes)

def bbox_rot90(bbox: BoxInternalType, factor: int, rows: int | None = None, cols: int | None = None) -> BoxInternalType:
    """Rotates a bounding box by 90 degrees CCW (see np.rot90)

    Args:
        bbox: A bounding box tuple (x_min, y_min, x_max, y_max).
        factor: Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        rows: Image rows.
        cols: Image cols.

    Returns:
        tuple: A bounding box tuple (x_min, y_min, x_max, y_max).

    """
    if factor not in {0, 1, 2, 3}:
        msg = "Parameter n must be in set {0, 1, 2, 3}"
        raise ValueError(msg)
    x_min, y_min, x_max, y_max = bbox[:4]
    if factor == 1:
        bbox = y_min, 1 - x_max, y_max, 1 - x_min
    elif factor == ROT90_180_FACTOR:
        bbox = 1 - x_max, 1 - y_max, 1 - x_min, 1 - y_min
    elif factor == ROT90_270_FACTOR:
        bbox = 1 - y_max, x_min, 1 - y_min, x_max
    return bbox

def smallest_max_size(img: np.ndarray, max_size: int, interpolation: int) -> np.ndarray:
    return _func_max_size(img, max_size, interpolation, min)

