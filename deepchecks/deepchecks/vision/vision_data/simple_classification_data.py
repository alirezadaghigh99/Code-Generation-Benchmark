def classification_dataset_from_directory(
        root: str,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
        object_type: Literal[VisionData, DataLoader] = 'DataLoader',
        **kwargs
) -> t.Union[t.Tuple[t.Union[DataLoader, VisionData]], t.Union[DataLoader, VisionData]]:
    """Load a simple classification dataset.

    The function expects that the data within the root folder
    to be structured one of the following ways:

        - root/
            - class1/
                image1.jpeg

        - root/
            - train/
                - class1/
                    image1.jpeg
            - test/
                - class1/
                    image1.jpeg

    Parameters
    ----------
    root : str
        path to the data
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
        type of the return value. If 'Dataset', :obj:`deepchecks.vision.VisionData`
        will be returned, otherwise :obj:`torch.utils.data.DataLoader`

    Returns
    -------
    t.Union[t.Tuple[t.Union[DataLoader, vision.ClassificationData]], t.Union[DataLoader, vision.ClassificationData]]
        A DataLoader or VisionDataset instance or tuple representing a single dataset or train and test datasets.
    """

    def batch_collate(batch):
        imgs, labels = zip(*batch)
        return list(imgs), list(labels)

    root_path = Path(root).absolute()
    if not (root_path.exists() and root_path.is_dir()):
        raise ValueError(f'{root_path} - path does not exist or is not a folder')

    roots_of_datasets = []
    if root_path.joinpath('train').exists():
        roots_of_datasets.append(root_path.joinpath('train'))
    if root_path.joinpath('test').exists():
        roots_of_datasets.append(root_path.joinpath('test'))
    if len(roots_of_datasets) == 0:
        roots_of_datasets.append(root_path)

    result = []
    for dataset_root in roots_of_datasets:
        dataset = SimpleClassificationDataset(root=str(dataset_root), **kwargs)
        if object_type == 'DataLoader':
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                    collate_fn=batch_collate, pin_memory=pin_memory, generator=torch.Generator())
            result.append(dataloader)
        elif object_type == 'VisionData':
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                    collate_fn=deepchecks_collate, pin_memory=pin_memory, generator=torch.Generator())
            result.append(VisionData(batch_loader=dataloader, label_map=dataset.reverse_classes_map,
                                     task_type='classification'))
        else:
            raise TypeError(f'Unknown value of object_type - {object_type}')
    return tuple(result) if len(result) > 1 else result[0]

