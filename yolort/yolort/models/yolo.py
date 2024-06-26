    def load_from_yolov5(
        cls,
        checkpoint_path: str,
        score_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        version: str = "r6.0",
        post_process: Optional[nn.Module] = None,
    ):
        """
        Load model state from the checkpoint trained by YOLOv5.

        Args:
            checkpoint_path (str): Path of the YOLOv5 checkpoint model.
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            version (str): upstream version released by the ultralytics/yolov5, Possible
                values are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        """
        model_info = load_from_ultralytics(checkpoint_path, version=version)
        backbone_name = f"darknet_{model_info['size']}_{version.replace('.', '_')}"
        depth_multiple = model_info["depth_multiple"]
        width_multiple = model_info["width_multiple"]
        use_p6 = model_info["use_p6"]
        backbone = darknet_pan_backbone(
            backbone_name, depth_multiple, width_multiple, version=version, use_p6=use_p6
        )
        model = cls(
            backbone,
            model_info["num_classes"],
            strides=model_info["strides"],
            anchor_grids=model_info["anchor_grids"],
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            post_process=post_process,
        )

        model.load_state_dict(model_info["state_dict"])
        return model