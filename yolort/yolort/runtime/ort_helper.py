def export_onnx(
    onnx_path: str,
    *,
    checkpoint_path: Optional[str] = None,
    model: Optional[nn.Module] = None,
    size: Tuple[int, int] = (640, 640),
    size_divisible: int = 32,
    score_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    version: str = "r6.0",
    skip_preprocess: bool = False,
    opset_version: int = 11,
    batch_size: int = 1,
    vanilla: bool = False,
    simplify: bool = False,
) -> None:
    """
    Export to ONNX models that can be used for ONNX Runtime inferencing.

    Args:
        onnx_path (string): The path to the ONNX graph to be exported.
        checkpoint_path (string, optional): Path of the custom trained YOLOv5 checkpoint.
            Default: None
        model (nn.Module): The defined PyTorch module to be exported. Default: None
        size: (Tuple[int, int]): the minimum and maximum size of the image to be rescaled.
            Default: (640, 640)
        size_divisible (int): Stride in the preprocessing. Default: 32
        score_thresh (float): Score threshold used for postprocessing the detections.
            Default: 0.25
        nms_thresh (float): NMS threshold used for postprocessing the detections. Default: 0.45
        version (string): Upstream YOLOv5 version. Default: 'r6.0'
        skip_preprocess (bool): Skip the preprocessing transformation when exporting the ONNX
            models. Default: False
        opset_version (int): Opset version for exporting ONNX models. Default: 11
        batch_size (int): Only used for models that include pre-processing, you need to specify
            the batch sizes and ensure that the number of input images is the same as the batches
            when inferring if you want to export multiple batches ONNX models. Default: 1
        vanilla (bool, optional): Whether to export a vanilla ONNX models. Default to False
        simplify (bool, optional): Whether to simplify the exported ONNX. Default to False
    """

    if vanilla:
        onnx_builder = VanillaONNXBuilder(
            checkpoint_path=checkpoint_path,
            score_thresh=score_thresh,
            iou_thresh=nms_thresh,
            opset_version=opset_version,
        )
    else:
        onnx_builder = ONNXBuilder(
            checkpoint_path=checkpoint_path,
            model=model,
            size=size,
            size_divisible=size_divisible,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            version=version,
            skip_preprocess=skip_preprocess,
            opset_version=opset_version,
            batch_size=batch_size,
        )

    onnx_builder.to_onnx(onnx_path, simplify)