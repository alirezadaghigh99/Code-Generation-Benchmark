def get_coco_segmentation_from_bool_mask(bool_mask):
    """
    Convert boolean mask to coco segmentation format
    [
        [x1, y1, x2, y2, x3, y3, ...],
        [x1, y1, x2, y2, x3, y3, ...],
        ...
    ]
    """
    # Generate polygons from mask
    mask = np.squeeze(bool_mask)
    mask = mask.astype(np.uint8)
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    polygons = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
    polygons = polygons[0] if len(polygons) == 2 else polygons[1]
    # Convert polygon to coco segmentation
    coco_segmentation = []
    for polygon in polygons:
        segmentation = polygon.flatten().tolist()
        # at least 3 points needed for a polygon
        if len(segmentation) >= 6:
            coco_segmentation.append(segmentation)
    return coco_segmentation

def exif_transpose(image: Image.Image) -> Image.Image:
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    Args:
        image (Image.Image): The image to transpose.

    Returns:
        Image.Image: The transposed image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image

def read_image(image_path: str):
    """
    Loads image as a numpy array from the given path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image as a numpy array.
    """
    # read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return image
    return image

