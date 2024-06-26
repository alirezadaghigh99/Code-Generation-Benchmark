def object_to_yolo(
    xyxy: np.ndarray,
    class_id: int,
    image_shape: Tuple[int, int, int],
    polygon: Optional[np.ndarray] = None,
) -> str:
    h, w, _ = image_shape
    if polygon is None:
        xyxy_relative = xyxy / np.array([w, h, w, h], dtype=np.float32)
        x_min, y_min, x_max, y_max = xyxy_relative
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        return f"{int(class_id)} {x_center:.5f} {y_center:.5f} {width:.5f} {height:.5f}"
    else:
        polygon_relative = polygon / np.array([w, h], dtype=np.float32)
        polygon_relative = polygon_relative.reshape(-1)
        polygon_parsed = " ".join([f"{value:.5f}" for value in polygon_relative])
        return f"{int(class_id)} {polygon_parsed}"