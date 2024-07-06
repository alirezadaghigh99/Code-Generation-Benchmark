def make_tmp_onnx_file(model: ModelProto) -> str:
    path = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
    save_onnx(model, path)
    return path

