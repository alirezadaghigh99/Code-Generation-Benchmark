def load_dataset(
        train: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = False,
        pin_memory: bool = True,
        object_type: Literal['VisionData', 'DataLoader'] = 'DataLoader',
        n_samples: t.Optional[int] = None,
        device: t.Union[str, torch.device] = 'cpu'
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
    shuffle : bool, default: False
        Whether to shuffle the dataset.
    pin_memory : bool, default: True
        If ``True``, the data loader will copy Tensors
        into CUDA pinned memory before returning them.
    object_type : Literal['Dataset', 'DataLoader'], default: 'DataLoader'
        type of the return value. If 'Dataset', :obj:`deepchecks.vision.VisionData`
        will be returned, otherwise :obj:`torch.utils.data.DataLoader`
    n_samples : int, optional
        Only relevant for loading a VisionData. Number of samples to load. Return the first n_samples if shuffle
        is False otherwise selects n_samples at random. If None, returns all samples.
    device : t.Union[str, torch.device], default : 'cpu'
        device to use in tensor calculations

    Returns
    -------
    Union[DataLoader, VisionData]
        A DataLoader or VisionData instance representing COCO128 dataset
    """
    coco_dir, dataset_name = CocoDataset.download_coco128(COCO_DIR)
    dataset = CocoDataset(root=str(coco_dir), name=dataset_name, train=train,
                          transforms=A.Compose([A.NoOp()], bbox_params=A.BboxParams(format='coco')))

    if object_type == 'DataLoader':
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=_batch_collate, pin_memory=pin_memory, generator=torch.Generator())
    elif object_type == 'VisionData':
        model = load_model(device=device)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                                collate_fn=deepchecks_collate(model), pin_memory=pin_memory,
                                generator=torch.Generator())
        dataloader = get_data_loader_sequential(dataloader, shuffle=shuffle, n_samples=n_samples)
        return VisionData(batch_loader=dataloader, label_map=LABEL_MAP, task_type='object_detection',
                          reshuffle_data=False)
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')

