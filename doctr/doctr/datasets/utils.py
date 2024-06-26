def crop_bboxes_from_image(img_path: Union[str, Path], geoms: np.ndarray) -> List[np.ndarray]:
    """Crop a set of bounding boxes from an image

    Args:
    ----
        img_path: path to the image
        geoms: a array of polygons of shape (N, 4, 2) or of straight boxes of shape (N, 4)

    Returns:
    -------
        a list of cropped images
    """
    with Image.open(img_path) as pil_img:
        img: np.ndarray = np.array(pil_img.convert("RGB"))
    # Polygon
    if geoms.ndim == 3 and geoms.shape[1:] == (4, 2):
        return extract_rcrops(img, geoms.astype(dtype=int))
    if geoms.ndim == 2 and geoms.shape[1] == 4:
        return extract_crops(img, geoms.astype(dtype=int))
    raise ValueError("Invalid geometry format")