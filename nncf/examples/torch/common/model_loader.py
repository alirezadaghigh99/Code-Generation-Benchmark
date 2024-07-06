def load_model(
    model, pretrained=True, num_classes=1000, model_params=None, weights_path: str = None
) -> torch.nn.Module:
    """

    ** WARNING: This is implemented using torch.load functionality,
    which itself uses Python's pickling facilities that may be used to perform
    arbitrary code execution during unpickling. Only load the data you trust.

    """
    logger.info("Loading model: {}".format(model))
    if model_params is None:
        model_params = {}
    if model in torchvision.models.__dict__:
        load_model_fn = partial(
            torchvision.models.__dict__[model], num_classes=num_classes, pretrained=pretrained, **model_params
        )
    elif model in custom_models.__dict__:
        load_model_fn = partial(
            custom_models.__dict__[model], num_classes=num_classes, pretrained=pretrained, **model_params
        )
    elif model == "mobilenet_v2_32x32":
        load_model_fn = partial(MobileNetV2For32x32, num_classes=100)
    else:
        raise Exception("Undefined model name")
    loaded_model = safe_thread_call(load_model_fn)
    if not pretrained and weights_path is not None:
        # Check if provided path is a url and download the checkpoint if yes
        if is_url(weights_path):
            weights_path = download_checkpoint(weights_path)
        sd = torch.load(weights_path, map_location="cpu", pickle_module=restricted_pickle_module)
        if MODEL_STATE_ATTR in sd:
            sd = sd[MODEL_STATE_ATTR]
        load_state(loaded_model, sd, is_resume=False)
    return loaded_model

