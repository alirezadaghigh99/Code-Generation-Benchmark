def cutout(
    img: np.ndarray,
    holes: Iterable[tuple[int, int, int, int]],
    fill_value: ColorType | Literal["random"] = 0,
) -> np.ndarray:
    """Apply cutout augmentation to the image by cutting out holes and filling them
    with either a given value or random noise.

    Args:
        img (np.ndarray): The image to augment.
        holes (Iterable[tuple[int, int, int, int]]): An iterable of tuples where each
            tuple contains the coordinates of the top-left and bottom-right corners of
            the rectangular hole (x1, y1, x2, y2).
        fill_value (Union[ColorType, Literal["random"]]): The fill value to use for the hole. Can be
            a single integer, a tuple or list of numbers for multichannel,
            or the string "random" to fill with random noise.

    Returns:
        np.ndarray: The augmented image.
    """
    img = img.copy()

    if isinstance(fill_value, (int, float, tuple, list)):
        fill_value = np.array(fill_value, dtype=img.dtype)

    for x1, y1, x2, y2 in holes:
        if isinstance(fill_value, str) and fill_value == "random":
            shape = (y2 - y1, x2 - x1) if img.ndim == MONO_CHANNEL_DIMENSIONS else (y2 - y1, x2 - x1, img.shape[2])
            random_fill = generate_random_fill(img.dtype, shape)
            img[y1:y2, x1:x2] = random_fill
        else:
            img[y1:y2, x1:x2] = fill_value

    return img

