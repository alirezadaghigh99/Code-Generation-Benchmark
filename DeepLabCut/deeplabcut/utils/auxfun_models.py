def check_for_weights(modeltype, parent_path):
    """gets local path to network weights and checks if they are present. If not, downloads them from tensorflow.org"""

    if modeltype not in MODELTYPE_FILEPATH_MAP.keys():
        print(
            "Currently ResNet (50, 101, 152), MobilenetV2 (1, 0.75, 0.5 and 0.35) and EfficientNet (b0-b6) are supported, please change 'resnet' entry in config.yaml!"
        )
        # Exit the function early if an unknown modeltype is provided.
        return parent_path

    exists = False
    model_path = parent_path / MODELTYPE_FILEPATH_MAP[modeltype]
    try:
        for file in os.listdir(model_path.parent):
            if model_path.name in file:
                exists = True
                break
    except FileNotFoundError:
        pass

    if not exists:
        if "efficientnet" in modeltype:
            download_weights(modeltype, model_path.parent)
        else:
            download_weights(modeltype, model_path)

    return str(model_path)

