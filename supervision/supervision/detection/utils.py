def contains_multiple_segments(
    mask: npt.NDArray[np.bool_], connectivity: int = 4
) -> bool:
    """
    Checks if the binary mask contains multiple unconnected foreground segments.

    Args:
        mask (npt.NDArray[np.bool_]): 2D binary mask where `True` indicates foreground
            object and `False` indicates background.
        connectivity (int) : Default: 4 is 4-way connectivity, which means that
            foreground pixels are the part of the same segment/component
            if their edges touch.
            Alternatively: 8 for 8-way connectivity, when foreground pixels are
            connected by their edges or corners touch.

    Returns:
        True when the mask contains multiple not connected components, False otherwise.

    Raises:
        ValueError: If connectivity(int) parameter value is not 4 or 8.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        mask = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0]
        ]).astype(bool)

        sv.contains_multiple_segments(mask=mask, connectivity=4)
        # True

        mask = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0]
        ]).astype(bool)

        sv.contains_multiple_segments(mask=mask, connectivity=4)
        # False
        ```

    ![contains_multiple_segments](https://media.roboflow.com/supervision-docs/contains-multiple-segments.png){ align=center width="800" }
    """  # noqa E501 // docs
    if connectivity != 4 and connectivity != 8:
        raise ValueError(
            "Incorrect connectivity value. Possible connectivity values: 4 or 8."
        )
    mask_uint8 = mask.astype(np.uint8)
    labels = np.zeros_like(mask_uint8, dtype=np.int32)
    number_of_labels, _ = cv2.connectedComponents(
        mask_uint8, labels, connectivity=connectivity
    )
    return number_of_labels > 2