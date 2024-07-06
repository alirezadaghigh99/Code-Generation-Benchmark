def inject_images_into_payload(
    payload: dict,
    encoded_images: List[Tuple[str, Optional[float]]],
    key: str = "image",
) -> dict:
    if len(encoded_images) == 0:
        return payload
    if len(encoded_images) > 1:
        images_payload = [
            {"type": "base64", "value": image} for image, _ in encoded_images
        ]
        payload[key] = images_payload
    else:
        payload[key] = {"type": "base64", "value": encoded_images[0][0]}
    return payload

