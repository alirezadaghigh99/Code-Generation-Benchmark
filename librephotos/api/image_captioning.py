def generate_caption(image_path, onnx, blip):
    json = {
        "image_path": image_path,
        "onnx": onnx,
        "blip": blip,
    }
    caption_response = requests.post(
        "http://localhost:8007/generate-caption", json=json
    ).json()

    return caption_response["caption"]

