def get(cls, model_name):
        """Get the model's metainfo by the model name.

        Args:
            model_name (str): The name of model.

        Returns:
            modelindex.models.Model: The metainfo of the specified model.
        """
        cls._register_mmpretrain_models()
        # lazy load config
        metainfo = copy.deepcopy(cls._models_dict.get(model_name.lower()))
        if metainfo is None:
            raise ValueError(
                f'Failed to find model "{model_name}". please use '
                '`mmpretrain.list_models` to get all available names.')
        if isinstance(metainfo.config, str):
            metainfo.config = Config.fromfile(metainfo.config)
        return metainfo

def get_model(model: Union[str, Config],
              pretrained: Union[str, bool] = False,
              device=None,
              device_map=None,
              offload_folder=None,
              url_mapping: Tuple[str, str] = None,
              **kwargs):
    """Get a pre-defined model or create a model from config.

    Args:
        model (str | Config): The name of model, the config file path or a
            config instance.
        pretrained (bool | str): When use name to specify model, you can
            use ``True`` to load the pre-defined pretrained weights. And you
            can also use a string to specify the path or link of weights to
            load. Defaults to False.
        device (str | torch.device | None): Transfer the model to the target
            device. Defaults to None.
        device_map (str | dict | None): A map that specifies where each
            submodule should go. It doesn't need to be refined to each
            parameter/buffer name, once a given module name is inside, every
            submodule of it will be sent to the same device. You can use
            `device_map="auto"` to automatically generate the device map.
            Defaults to None.
        offload_folder (str | None): If the `device_map` contains any value
            `"disk"`, the folder where we will offload weights.
        url_mapping (Tuple[str, str], optional): The mapping of pretrained
            checkpoint link. For example, load checkpoint from a local dir
            instead of download by ``('https://.*/', './checkpoint')``.
            Defaults to None.
        **kwargs: Other keyword arguments of the model config.

    Returns:
        mmengine.model.BaseModel: The result model.

    Examples:
        Get a ResNet-50 model and extract images feature:

        >>> import torch
        >>> from mmpretrain import get_model
        >>> inputs = torch.rand(16, 3, 224, 224)
        >>> model = get_model('resnet50_8xb32_in1k', pretrained=True, backbone=dict(out_indices=(0, 1, 2, 3)))
        >>> feats = model.extract_feat(inputs)
        >>> for feat in feats:
        ...     print(feat.shape)
        torch.Size([16, 256])
        torch.Size([16, 512])
        torch.Size([16, 1024])
        torch.Size([16, 2048])

        Get Swin-Transformer model with pre-trained weights and inference:

        >>> from mmpretrain import get_model, inference_model
        >>> model = get_model('swin-base_16xb64_in1k', pretrained=True)
        >>> result = inference_model(model, 'demo/demo.JPEG')
        >>> print(result['pred_class'])
        'sea snake'
    """  # noqa: E501
    if device_map is not None:
        from .utils import dispatch_model
        dispatch_model._verify_require()

    metainfo = None
    if isinstance(model, Config):
        config = copy.deepcopy(model)
        if pretrained is True and 'load_from' in config:
            pretrained = config.load_from
    elif isinstance(model, (str, PathLike)) and Path(model).suffix == '.py':
        config = Config.fromfile(model)
        if pretrained is True and 'load_from' in config:
            pretrained = config.load_from
    elif isinstance(model, str):
        metainfo = ModelHub.get(model)
        config = metainfo.config
        if pretrained is True and metainfo.weights is not None:
            pretrained = metainfo.weights
    else:
        raise TypeError('model must be a name, a path or a Config object, '
                        f'but got {type(config)}')

    if pretrained is True:
        warnings.warn('Unable to find pre-defined checkpoint of the model.')
        pretrained = None
    elif pretrained is False:
        pretrained = None

    if kwargs:
        config.merge_from_dict({'model': kwargs})
    config.model.setdefault('data_preprocessor',
                            config.get('data_preprocessor', None))

    from mmengine.registry import DefaultScope

    from mmpretrain.registry import MODELS
    with DefaultScope.overwrite_default_scope('mmpretrain'):
        model = MODELS.build(config.model)

    dataset_meta = {}
    if pretrained:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        from mmengine.runner import load_checkpoint
        if url_mapping is not None:
            pretrained = re.sub(url_mapping[0], url_mapping[1], pretrained)
        checkpoint = load_checkpoint(model, pretrained, map_location='cpu')
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmpretrain 1.x
            dataset_meta = checkpoint['meta']['dataset_meta']
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # mmcls 0.x
            dataset_meta = {'classes': checkpoint['meta']['CLASSES']}

    if len(dataset_meta) == 0 and 'test_dataloader' in config:
        from mmpretrain.registry import DATASETS
        dataset_class = DATASETS.get(config.test_dataloader.dataset.type)
        dataset_meta = getattr(dataset_class, 'METAINFO', {})

    if device_map is not None:
        model = dispatch_model(
            model, device_map=device_map, offload_folder=offload_folder)
    elif device is not None:
        model.to(device)

    model._dataset_meta = dataset_meta  # save the dataset meta
    model._config = config  # save the config in the model
    model._metainfo = metainfo  # save the metainfo in the model
    model.eval()
    return model

