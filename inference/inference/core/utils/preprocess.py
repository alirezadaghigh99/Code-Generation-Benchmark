def take_static_crop(image: np.ndarray, crop_parameters: Dict[str, int]) -> np.ndarray:
    height, width = image.shape[0:2]
    x_min = int(crop_parameters["x_min"] / 100 * width)
    y_min = int(crop_parameters["y_min"] / 100 * height)
    x_max = int(crop_parameters["x_max"] / 100 * width)
    y_max = int(crop_parameters["y_max"] / 100 * height)
    return image[y_min:y_max, x_min:x_max, :]