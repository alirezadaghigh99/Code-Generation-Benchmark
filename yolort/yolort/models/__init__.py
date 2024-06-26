def yolov5s(upstream_version: str = "r6.0", export_friendly: bool = False, **kwargs: Any):
    """
    Args:
        upstream_version (str): model released by the upstream YOLOv5. Possible values
            are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        export_friendly (bool): Deciding whether to use (ONNX/TVM) export friendly mode.
            Default: False.
    """
    if upstream_version == "r3.1":
        model = YOLOv5(arch="yolov5_darknet_pan_s_r31", **kwargs)
    elif upstream_version == "r4.0":
        model = YOLOv5(arch="yolov5_darknet_pan_s_r40", **kwargs)
    elif upstream_version == "r6.0":
        model = YOLOv5(arch="yolov5_darknet_pan_s_r60", **kwargs)
    else:
        raise NotImplementedError("Currently doesn't support this versions.")

    if export_friendly:
        _export_module_friendly(model)

    return model