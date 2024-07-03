def extract_backbone(path: str) -> tuple[str, 'OrderedDict[str, Tensor]']:
    """Extracts a backbone from a lightning checkpoint file.

    Args:
        path: path to checkpoint file (.ckpt)

    Returns:
        tuple containing model name and state dict

    Raises:
        ValueError: if 'model' or 'backbone' not in
            checkpoint['hyper_parameters']

    .. versionchanged:: 0.4
        Renamed from *extract_encoder* to *extract_backbone*
    """
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    if 'model' in checkpoint['hyper_parameters']:
        name = checkpoint['hyper_parameters']['model']
        state_dict = checkpoint['state_dict']
        state_dict = OrderedDict({k: v for k, v in state_dict.items() if 'model.' in k})
        state_dict = OrderedDict(
            {k.replace('model.', ''): v for k, v in state_dict.items()}
        )
    elif 'backbone' in checkpoint['hyper_parameters']:
        name = checkpoint['hyper_parameters']['backbone']
        state_dict = checkpoint['state_dict']
        state_dict = OrderedDict(
            {k: v for k, v in state_dict.items() if 'model.backbone.model' in k}
        )
        state_dict = OrderedDict(
            {k.replace('model.backbone.model.', ''): v for k, v in state_dict.items()}
        )
    else:
        raise ValueError(
            'Unknown checkpoint task. Only backbone or model extraction is supported'
        )

    return name, state_dict