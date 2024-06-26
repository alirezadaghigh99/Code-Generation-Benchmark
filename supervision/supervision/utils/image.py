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