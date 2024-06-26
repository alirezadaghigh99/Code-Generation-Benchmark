def convert_to_relative_coords(geoms: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """Convert a geometry to relative coordinates

    Args:
    ----
        geoms: a set of polygons of shape (N, 4, 2) or of straight boxes of shape (N, 4)
        img_shape: the height and width of the image

    Returns:
    -------
        the updated geometry
    """
    # Polygon
    if geoms.ndim == 3 and geoms.shape[1:] == (4, 2):
        polygons: np.ndarray = np.empty(geoms.shape, dtype=np.float32)
        polygons[..., 0] = geoms[..., 0] / img_shape[1]
        polygons[..., 1] = geoms[..., 1] / img_shape[0]
        return polygons.clip(0, 1)
    if geoms.ndim == 2 and geoms.shape[1] == 4:
        boxes: np.ndarray = np.empty(geoms.shape, dtype=np.float32)
        boxes[:, ::2] = geoms[:, ::2] / img_shape[1]
        boxes[:, 1::2] = geoms[:, 1::2] / img_shape[0]
        return boxes.clip(0, 1)

    raise ValueError(f"invalid format for arg `geoms`: {geoms.shape}")