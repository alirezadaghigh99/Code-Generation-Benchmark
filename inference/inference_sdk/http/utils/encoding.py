def bytes_to_pillow_image(payload: bytes) -> Image.Image:
    buffer = BytesIO(payload)
    try:
        return Image.open(buffer)
    except UnidentifiedImageError as error:
        raise EncodingError("Could not encode bytes to PIL image.") from error