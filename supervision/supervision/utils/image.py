def create_tiles(
    images: List[ImageType],
    grid_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
    single_tile_size: Optional[Tuple[int, int]] = None,
    tile_scaling: Literal["min", "max", "avg"] = "avg",
    tile_padding_color: Union[Tuple[int, int, int], Color] = Color.from_hex("#D9D9D9"),
    tile_margin: int = 10,
    tile_margin_color: Union[Tuple[int, int, int], Color] = Color.from_hex("#BFBEBD"),
    return_type: Literal["auto", "cv2", "pillow"] = "auto",
    titles: Optional[List[Optional[str]]] = None,
    titles_anchors: Optional[Union[Point, List[Optional[Point]]]] = None,
    titles_color: Union[Tuple[int, int, int], Color] = Color.from_hex("#262523"),
    titles_scale: Optional[float] = None,
    titles_thickness: int = 1,
    titles_padding: int = 10,
    titles_text_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    titles_background_color: Union[Tuple[int, int, int], Color] = Color.from_hex(
        "#D9D9D9"
    ),
    default_title_placement: RelativePosition = "top",
) -> ImageType:
    """
    Creates tiles mosaic from input images, automating grid placement and
    converting images to common resolution maintaining aspect ratio. It is
    also possible to render text titles on tiles, using optional set of
    parameters specifying text drawing (see parameters description).

    Automated grid placement will try to maintain square shape of grid
    (with size being the nearest integer square root of #images), up to two exceptions:
    * if there are up to 3 images - images will be displayed in single row
    * if square-grid placement causes last row to be empty - number of rows is trimmed
        until last row has at least one image

    Args:
        images (List[ImageType]): Images to create tiles. Elements can be either
            np.ndarray or PIL.Image, common representation will be agreed by the
            function.
        grid_size (Optional[Tuple[Optional[int], Optional[int]]]): Expected grid
            size in format (n_rows, n_cols). If not given - automated grid placement
            will be applied. One may also provide only one out of two elements of the
            tuple - then grid will be created with either n_rows or n_cols fixed,
            leaving the other dimension to be adjusted by the number of images
        single_tile_size (Optional[Tuple[int, int]]): sizeof a single tile element
            provided in (width, height) format. If not given - size of tile will be
            automatically calculated based on `tile_scaling` parameter.
        tile_scaling (Literal["min", "max", "avg"]): If `single_tile_size` is not
            given - parameter will be used to calculate tile size - using
            min / max / avg size of image provided in `images` list.
        tile_padding_color (Union[Tuple[int, int, int], sv.Color]): Color to be used in
            images letterbox procedure (while standardising tiles sizes) as a padding.
            If tuple provided - should be BGR.
        tile_margin (int): size of margin between tiles (in pixels)
        tile_margin_color (Union[Tuple[int, int, int], sv.Color]): Color of tile margin.
            If tuple provided - should be BGR.
        return_type (Literal["auto", "cv2", "pillow"]): Parameter dictates the format of
            return image. One may choose specific type ("cv2" or "pillow") to enforce
            conversion. "auto" mode takes a majority vote between types of elements in
            `images` list - resolving draws in favour of OpenCV format. "auto" can be
            safely used when all input images are of the same type.
        titles (Optional[List[Optional[str]]]): Optional titles to be added to tiles.
            Elements of that list may be empty - then specific tile (in order presented
            in `images` parameter) will not be filled with title. It is possible to
            provide list of titles shorter than `images` - then remaining titles will
            be assumed empty.
        titles_anchors (Optional[Union[Point, List[Optional[Point]]]]): Parameter to
            specify anchor points for titles. It is possible to specify anchor either
            globally or for specific tiles (following order of `images`).
            If not given (either globally, or for specific element of the list),
            it will be calculated automatically based on `default_title_placement`.
        titles_color (Union[Tuple[int, int, int], Color]): Color of titles text.
            If tuple provided - should be BGR.
        titles_scale (Optional[float]): Scale of titles. If not provided - value will
            be calculated using `calculate_optimal_text_scale(...)`.
        titles_thickness (int): Thickness of titles text.
        titles_padding (int): Size of titles padding.
        titles_text_font (int): Font to be used to render titles. Must be integer
            constant representing OpenCV font.
            (See docs: https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html)
        titles_background_color (Union[Tuple[int, int, int], Color]): Color of title
            text padding.
        default_title_placement (Literal["top", "bottom"]): Parameter specifies title
            anchor placement in case if explicit anchor is not provided.

    Returns:
        ImageType: Image with all input images located in tails grid. The output type is
            determined by `return_type` parameter.

    Raises:
        ValueError: In case when input images list is empty, provided `grid_size` is too
            small to fit all images, `tile_scaling` mode is invalid.
    """
    if len(images) == 0:
        raise ValueError("Could not create image tiles from empty list of images.")
    if return_type == "auto":
        return_type = _negotiate_tiles_format(images=images)
    tile_padding_color = unify_to_bgr(color=tile_padding_color)
    tile_margin_color = unify_to_bgr(color=tile_margin_color)
    images = images_to_cv2(images=images)
    if single_tile_size is None:
        single_tile_size = _aggregate_images_shape(images=images, mode=tile_scaling)
    resized_images = [
        letterbox_image(
            image=i, resolution_wh=single_tile_size, color=tile_padding_color
        )
        for i in images
    ]
    grid_size = _establish_grid_size(images=images, grid_size=grid_size)
    if len(images) > grid_size[0] * grid_size[1]:
        raise ValueError(
            f"Could not place {len(images)} in grid with size: {grid_size}."
        )
    if titles is not None:
        titles = fill(sequence=titles, desired_size=len(images), content=None)
    titles_anchors = (
        [titles_anchors]
        if not issubclass(type(titles_anchors), list)
        else titles_anchors
    )
    titles_anchors = fill(
        sequence=titles_anchors, desired_size=len(images), content=None
    )
    titles_color = unify_to_bgr(color=titles_color)
    titles_background_color = unify_to_bgr(color=titles_background_color)
    tiles = _generate_tiles(
        images=resized_images,
        grid_size=grid_size,
        single_tile_size=single_tile_size,
        tile_padding_color=tile_padding_color,
        tile_margin=tile_margin,
        tile_margin_color=tile_margin_color,
        titles=titles,
        titles_anchors=titles_anchors,
        titles_color=titles_color,
        titles_scale=titles_scale,
        titles_thickness=titles_thickness,
        titles_padding=titles_padding,
        titles_text_font=titles_text_font,
        titles_background_color=titles_background_color,
        default_title_placement=default_title_placement,
    )
    if return_type == "pillow":
        tiles = cv2_to_pillow(image=tiles)
    return tiles

def resize_image(
    image: ImageType,
    resolution_wh: Tuple[int, int],
    keep_aspect_ratio: bool = False,
) -> ImageType:
    """
    Resizes the given image to a specified resolution. Can maintain the original aspect
    ratio or resize directly to the desired dimensions.

    Args:
        image (ImageType): The image to be resized. `ImageType` is a flexible type,
            accepting either `numpy.ndarray` or `PIL.Image.Image`.
        resolution_wh (Tuple[int, int]): The target resolution as
            `(width, height)`.
        keep_aspect_ratio (bool, optional): Flag to maintain the image's original
            aspect ratio. Defaults to `False`.

    Returns:
        (ImageType): The resized image. The type is determined by the input type and
            may be either a `numpy.ndarray` or `PIL.Image.Image`.

    === "OpenCV"

        ```python
        import cv2
        import supervision as sv

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        image.shape
        # (1080, 1920, 3)

        resized_image = sv.resize_image(
            image=image, resolution_wh=(1000, 1000), keep_aspect_ratio=True
        )
        resized_image.shape
        # (562, 1000, 3)
        ```

    === "Pillow"

        ```python
        from PIL import Image
        import supervision as sv

        image = Image.open(<SOURCE_IMAGE_PATH>)
        image.size
        # (1920, 1080)

        resized_image = sv.resize_image(
            image=image, resolution_wh=(1000, 1000), keep_aspect_ratio=True
        )
        resized_image.size
        # (1000, 562)
        ```

    ![resize_image](https://media.roboflow.com/supervision-docs/resize-image.png){ align=center width="800" }
    """  # noqa E501 // docs
    if keep_aspect_ratio:
        image_ratio = image.shape[1] / image.shape[0]
        target_ratio = resolution_wh[0] / resolution_wh[1]
        if image_ratio >= target_ratio:
            width_new = resolution_wh[0]
            height_new = int(resolution_wh[0] / image_ratio)
        else:
            height_new = resolution_wh[1]
            width_new = int(resolution_wh[1] * image_ratio)
    else:
        width_new, height_new = resolution_wh

    return cv2.resize(image, (width_new, height_new), interpolation=cv2.INTER_LINEAR)

def letterbox_image(
    image: ImageType,
    resolution_wh: Tuple[int, int],
    color: Union[Tuple[int, int, int], Color] = Color.BLACK,
) -> ImageType:
    """
    Resizes and pads an image to a specified resolution with a given color, maintaining
    the original aspect ratio.

    Args:
        image (ImageType): The image to be resized. `ImageType` is a flexible type,
            accepting either `numpy.ndarray` or `PIL.Image.Image`.
        resolution_wh (Tuple[int, int]): The target resolution as
            `(width, height)`.
        color (Union[Tuple[int, int, int], Color]): The color to pad with. If tuple
            provided it should be in BGR format.

    Returns:
        (ImageType): The resized image. The type is determined by the input type and
            may be either a `numpy.ndarray` or `PIL.Image.Image`.

    === "OpenCV"

        ```python
        import cv2
        import supervision as sv

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        image.shape
        # (1080, 1920, 3)

        letterboxed_image = sv.letterbox_image(image=image, resolution_wh=(1000, 1000))
        letterboxed_image.shape
        # (1000, 1000, 3)
        ```

    === "Pillow"

        ```python
        from PIL import Image
        import supervision as sv

        image = Image.open(<SOURCE_IMAGE_PATH>)
        image.size
        # (1920, 1080)

        letterboxed_image = sv.letterbox_image(image=image, resolution_wh=(1000, 1000))
        letterboxed_image.size
        # (1000, 1000)
        ```

    ![letterbox_image](https://media.roboflow.com/supervision-docs/letterbox-image.png){ align=center width="800" }
    """  # noqa E501 // docs
    color = unify_to_bgr(color=color)
    resized_image = resize_image(
        image=image, resolution_wh=resolution_wh, keep_aspect_ratio=True
    )
    height_new, width_new = resized_image.shape[:2]
    padding_top = (resolution_wh[1] - height_new) // 2
    padding_bottom = resolution_wh[1] - height_new - padding_top
    padding_left = (resolution_wh[0] - width_new) // 2
    padding_right = resolution_wh[0] - width_new - padding_left
    return cv2.copyMakeBorder(
        resized_image,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

