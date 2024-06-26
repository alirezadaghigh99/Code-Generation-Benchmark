def rle_to_mask(
    rle: Union[npt.NDArray[np.int_], List[int]], resolution_wh: Tuple[int, int]
) -> npt.NDArray[np.bool_]:
    """
    Converts run-length encoding (RLE) to a binary mask.

    Args:
        rle (Union[npt.NDArray[np.int_], List[int]]): The 1D RLE array, the format
            used in the COCO dataset (column-wise encoding, values of an array with
            even indices represent the number of pixels assigned as background,
            values of an array with odd indices represent the number of pixels
            assigned as foreground object).
        resolution_wh (Tuple[int, int]): The width (w) and height (h)
            of the desired binary mask.

    Returns:
        The generated 2D Boolean mask of shape `(h, w)`, where the foreground object is
            marked with `True`'s and the rest is filled with `False`'s.

    Raises:
        AssertionError: If the sum of pixels encoded in RLE differs from the
            number of pixels in the expected mask (computed based on resolution_wh).

    Examples:
        ```python
        import supervision as sv

        sv.rle_to_mask([5, 2, 2, 2, 5], (4, 4))
        # array([
        #     [False, False, False, False],
        #     [False, True,  True,  False],
        #     [False, True,  True,  False],
        #     [False, False, False, False],
        # ])
        ```
    """
    if isinstance(rle, list):
        rle = np.array(rle, dtype=int)

    width, height = resolution_wh

    assert width * height == np.sum(rle), (
        "the sum of the number of pixels in the RLE must be the same "
        "as the number of pixels in the expected mask"
    )

    zero_one_values = np.zeros(shape=(rle.size, 1), dtype=np.uint8)
    zero_one_values[1::2] = 1

    decoded_rle = np.repeat(zero_one_values, rle, axis=0)
    decoded_rle = np.append(
        decoded_rle, np.zeros(width * height - len(decoded_rle), dtype=np.uint8)
    )
    return decoded_rle.reshape((height, width), order="F")