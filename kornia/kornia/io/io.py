def load_image(path_file: str | Path, desired_type: ImageLoadType, device: Device = "cpu") -> Tensor:
    """Read an image file and decode using the Kornia Rust backend.

    Args:
        path_file: Path to a valid image file.
        desired_type: the desired image type, defined by color space and dtype.
        device: the device where you want to get your image placed.

    Return:
        Image tensor with shape :math:`(3,H,W)`.
    """
    if not isinstance(path_file, Path):
        path_file = Path(path_file)

    # read the image using the kornia_rs package
    image: Tensor = _load_image_to_tensor(path_file, device)  # CxHxW

    if desired_type == ImageLoadType.UNCHANGED:
        return image
    elif desired_type == ImageLoadType.GRAY8:
        if image.shape[0] == 1 and image.dtype == torch.uint8:
            return image
        elif image.shape[0] == 3 and image.dtype == torch.uint8:
            gray8 = rgb_to_grayscale(image)
            return gray8
        elif image.shape[0] == 4 and image.dtype == torch.uint8:
            gray32 = rgb_to_grayscale(rgba_to_rgb(_to_float32(image)))
            return _to_uint8(gray32)

    elif desired_type == ImageLoadType.RGB8:
        if image.shape[0] == 3 and image.dtype == torch.uint8:
            return image
        elif image.shape[0] == 1 and image.dtype == torch.uint8:
            rgb8 = grayscale_to_rgb(image)
            return rgb8

    elif desired_type == ImageLoadType.RGBA8:
        if image.shape[0] == 3 and image.dtype == torch.uint8:
            rgba32 = rgb_to_rgba(_to_float32(image), 0.0)
            return _to_uint8(rgba32)

    elif desired_type == ImageLoadType.GRAY32:
        if image.shape[0] == 1 and image.dtype == torch.uint8:
            return _to_float32(image)
        elif image.shape[0] == 3 and image.dtype == torch.uint8:
            gray32 = rgb_to_grayscale(_to_float32(image))
            return gray32
        elif image.shape[0] == 4 and image.dtype == torch.uint8:
            gray32 = rgb_to_grayscale(rgba_to_rgb(_to_float32(image)))
            return gray32

    elif desired_type == ImageLoadType.RGB32:
        if image.shape[0] == 3 and image.dtype == torch.uint8:
            return _to_float32(image)
        elif image.shape[0] == 1 and image.dtype == torch.uint8:
            rgb32 = grayscale_to_rgb(_to_float32(image))
            return rgb32

    raise NotImplementedError(f"Unknown type: {desired_type}")

