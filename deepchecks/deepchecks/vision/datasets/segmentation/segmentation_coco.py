def load_dataset(
        train: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
        object_type: Literal['VisionData', 'DataLoader'] = 'VisionData',
        test_mode: bool = False
) -> t.Union[DataLoader, VisionData]:
    """Get the COCO128 dataset and return a dataloader.

    Parameters
    ----------
    train : bool, default: True
        if `True` train dataset, otherwise test dataset
    batch_size : int, default: 32
        Batch size for the dataloader.
    num_workers : int, default: 0
        Number of workers for the dataloader.
    shuffle : bool, default: True
        Whether to shuffle the dataset.
    pin_memory : bool, default: True
        If ``True``, the data loader will copy Tensors
        into CUDA pinned memory before returning them.
    object_type : Literal['Dataset', 'DataLoader'], default: 'DataLoader'
        type of the return value. If 'Dataset', :obj:`deepchecks.vision.VisionDataset`
        will be returned, otherwise :obj:`torch.utils.data.DataLoader`
    test_mode: bool, default False
        whether to load this dataset in "test_mode", meaning very minimal number of images in order to use for
        unittests.

    Returns
    -------
    Union[DataLoader, VisionDataset]

        A DataLoader or VisionDataset instance representing COCO128 dataset
    """
    root = DATA_DIR
    dataset = CocoSegmentationDataset.load_or_download(root=root, train=train, test_mode=test_mode)

    if object_type == 'DataLoader':
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=_batch_collate, pin_memory=pin_memory, generator=torch.Generator())
    elif object_type == 'VisionData':
        model = load_model()
        loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=deepchecks_collate(model), pin_memory=pin_memory, generator=torch.Generator())
        loader = get_data_loader_sequential(loader, shuffle=shuffle)
        return VisionData(batch_loader=loader, task_type='semantic_segmentation', label_map=LABEL_MAP,
                          reshuffle_data=False)
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')

