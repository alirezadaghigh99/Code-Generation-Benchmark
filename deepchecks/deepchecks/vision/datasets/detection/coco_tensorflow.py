def load_dataset(
        train: bool = True,
        shuffle: bool = False,
        object_type: Literal['VisionData', 'Dataset'] = 'Dataset',
        n_samples: t.Optional[int] = None,
) -> t.Union[tf.data.Dataset, vision.VisionData]:
    """Get the COCO128 dataset and return a dataloader.

    Parameters
    ----------
    train : bool, default: True
        if `True` train dataset, otherwise test dataset
    shuffle : bool, default: False
        Whether to shuffle the dataset.
    object_type : Literal['Dataset', 'Dataset'], default: 'Dataset'
        type of the return value. If 'Dataset', :obj:`deepchecks.vision.VisionData`
        will be returned, otherwise :obj:`tf.data.Dataset`.
    n_samples : int, optional
        Number of samples to load. Return the first n_samples if shuffle
        is False otherwise selects n_samples at random. If None, returns all samples.

    Returns
    -------
    Union[Dataset, VisionData]
        A Dataset or VisionData instance representing COCO128 dataset
    """
    transforms = A.Compose([A.NoOp()], bbox_params=A.BboxParams(format='coco'))
    coco_dataset = create_tf_dataset(train, n_samples, transforms)
    if shuffle:
        coco_dataset = coco_dataset.shuffle(128)

    if object_type == 'Dataset':
        return coco_dataset
    elif object_type == 'VisionData':
        model = hub.load(_MODEL_URL)
        coco_dataset = coco_dataset.map(deepchecks_map(model))
        return VisionData(batch_loader=coco_dataset, label_map=LABEL_MAP, task_type='object_detection',
                          reshuffle_data=False)
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')

