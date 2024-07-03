def load_from_ultralytics(checkpoint_path: str, version: str = "r6.0"):
    """
    Allows the user to load model state file from the checkpoint trained from
    the ultralytics/yolov5.

    Args:
        checkpoint_path (str): Path of the YOLOv5 checkpoint model.
        version (str): upstream version released by the ultralytics/yolov5, Possible
            values are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
    """

    if version not in ["r3.1", "r4.0", "r6.0"]:
        raise NotImplementedError(
            f"Currently does not support version: {version}. Feel free to file an issue "
            "labeled enhancement to us."
        )

    checkpoint_yolov5 = load_yolov5_model(checkpoint_path)
    num_classes = checkpoint_yolov5.yaml["nc"]
    strides = checkpoint_yolov5.stride
    # YOLOv5 will change the anchors setting when using the auto-anchor mechanism. So we
    # use the following formula to compute the anchor_grids instead of attaching it via
    # checkpoint_yolov5.yaml["anchors"]
    num_anchors = checkpoint_yolov5.model[-1].anchors.shape[1]
    anchor_grids = (
        (checkpoint_yolov5.model[-1].anchors * checkpoint_yolov5.model[-1].stride.view(-1, 1, 1))
        .reshape(1, -1, 2 * num_anchors)
        .tolist()[0]
    )

    depth_multiple = checkpoint_yolov5.yaml["depth_multiple"]
    width_multiple = checkpoint_yolov5.yaml["width_multiple"]

    use_p6 = False
    if len(strides) == 4:
        use_p6 = True

    if use_p6:
        inner_block_maps = {"0": "11", "1": "12", "3": "15", "4": "16", "6": "19", "7": "20"}
        layer_block_maps = {"0": "23", "1": "24", "2": "26", "3": "27", "4": "29", "5": "30", "6": "32"}
        p6_block_maps = {"0": "9", "1": "10"}
        head_ind = 33
        head_name = "m"
    else:
        inner_block_maps = {"0": "9", "1": "10", "3": "13", "4": "14"}
        layer_block_maps = {"0": "17", "1": "18", "2": "20", "3": "21", "4": "23"}
        p6_block_maps = None
        head_ind = 24
        head_name = "m"

    convert_yolo_checkpoint = CheckpointConverter(
        depth_multiple,
        width_multiple,
        inner_block_maps=inner_block_maps,
        layer_block_maps=layer_block_maps,
        p6_block_maps=p6_block_maps,
        strides=strides,
        anchor_grids=anchor_grids,
        head_ind=head_ind,
        head_name=head_name,
        num_classes=num_classes,
        version=version,
        use_p6=use_p6,
    )
    convert_yolo_checkpoint.updating(checkpoint_yolov5)
    state_dict = convert_yolo_checkpoint.model.half().state_dict()

    size = get_yolov5_size(depth_multiple, width_multiple)

    return {
        "num_classes": num_classes,
        "depth_multiple": depth_multiple,
        "width_multiple": width_multiple,
        "strides": strides,
        "anchor_grids": anchor_grids,
        "use_p6": use_p6,
        "size": size,
        "state_dict": state_dict,
    }