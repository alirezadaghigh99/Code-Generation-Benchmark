def load_image_from_string(
    reference: str,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Tuple[str, Optional[float]]:
    if uri_is_http_link(uri=reference):
        return load_image_from_url(
            url=reference, max_height=max_height, max_width=max_width
        )
    if os.path.exists(reference):
        if max_height is None or max_width is None:
            with open(reference, "rb") as f:
                img_bytes = f.read()
            img_base64_str = encode_base_64(payload=img_bytes)
            return img_base64_str, None
        local_image = cv2.imread(reference)
        if local_image is None:
            raise EncodingError(f"Could not load image from {reference}")
        local_image, scaling_factor = resize_opencv_image(
            image=local_image,
            max_height=max_height,
            max_width=max_width,
        )
        return numpy_array_to_base64_jpeg(image=local_image), scaling_factor
    if max_height is not None and max_width is not None:
        image_bytes = base64.b64decode(reference)
        image = bytes_to_opencv_image(payload=image_bytes)
        image, scaling_factor = resize_opencv_image(
            image=image,
            max_height=max_height,
            max_width=max_width,
        )
        return numpy_array_to_base64_jpeg(image=image), scaling_factor
    return reference, None

def uri_is_http_link(uri: str) -> bool:
    return uri.startswith("http://") or uri.startswith("https://")

