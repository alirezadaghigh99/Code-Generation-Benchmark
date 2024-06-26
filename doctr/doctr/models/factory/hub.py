def from_hub(repo_id: str, **kwargs: Any):
    """Instantiate & load a pretrained model from HF hub.

    >>> from doctr.models import from_hub
    >>> model = from_hub("mindee/fasterrcnn_mobilenet_v3_large_fpn")

    Args:
    ----
        repo_id: HuggingFace model hub repo
        kwargs: kwargs of `hf_hub_download` or `snapshot_download`

    Returns:
    -------
        Model loaded with the checkpoint
    """
    # Get the config
    with open(hf_hub_download(repo_id, filename="config.json", **kwargs), "rb") as f:
        cfg = json.load(f)

    arch = cfg["arch"]
    task = cfg["task"]
    cfg.pop("arch")
    cfg.pop("task")

    if task == "classification":
        model = models.classification.__dict__[arch](
            pretrained=False, classes=cfg["classes"], num_classes=cfg["num_classes"]
        )
    elif task == "detection":
        model = models.detection.__dict__[arch](pretrained=False)
    elif task == "recognition":
        model = models.recognition.__dict__[arch](pretrained=False, input_shape=cfg["input_shape"], vocab=cfg["vocab"])

    # update model cfg
    model.cfg = cfg

    # Load checkpoint
    if is_torch_available():
        state_dict = torch.load(hf_hub_download(repo_id, filename="pytorch_model.bin", **kwargs), map_location="cpu")
        model.load_state_dict(state_dict)
    else:  # tf
        repo_path = snapshot_download(repo_id, **kwargs)
        model.load_weights(os.path.join(repo_path, "tf_model", "weights"))

    return model