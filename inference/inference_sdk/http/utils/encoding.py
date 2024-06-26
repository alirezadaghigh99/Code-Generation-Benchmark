def encode_base_64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("utf-8")